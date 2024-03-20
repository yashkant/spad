import numpy as np
import torch
import time
import imageio
from skimage.draw import line
from easydict import EasyDict as edict

from pytorch3d.renderer import NDCMultinomialRaysampler, ray_bundle_to_ray_points
from pytorch3d.utils import cameras_from_opencv_projection
from einops import rearrange

from torch.nn import functional as F

# cache for fast epipolar line drawing
try:
    masks32 = np.load("/fs01/home/yashkant/spad-code/cache/masks32.npy", allow_pickle=True)
except:
    print(f"failed to load cache for fast epipolar line drawing, this does not affect final results")
    masks32 = None


def compute_epipolar_mask(src_frame, tgt_frame, imh, imw, dialate_mask=True, debug_depth=False, visualize_mask=False):
    """
    src_frame: source frame containing camera
    tgt_frame: target frame containing camera
    debug_depth: if True, uses depth map to compute epipolar lines on target image (debugging)
    visualize_mask: if True, saves a batched attention masks (debugging)
    """

    # generates raybundle using camera intrinsics and extrinsics
    src_ray_bundle = NDCMultinomialRaysampler(
        image_width=imw,
        image_height=imh,
        n_pts_per_ray=1,
        min_depth=1.0,
        max_depth=1.0,
    )(src_frame.camera)
    
    src_depth = getattr(src_frame, "depth_map", None)
    if debug_depth and src_depth is not None:
        src_depth = src_depth[:, 0, ..., None]
        src_depth[src_depth >= 100] = 100 # clip depth
    else:
        # get points in world space (at fixed depth)
        src_depth = 3.5 * torch.ones((1, imh, imw, 1), dtype=torch.float32, device=src_frame.camera.device)

    pts_world = ray_bundle_to_ray_points(
      src_ray_bundle._replace(lengths=src_depth)
    ).squeeze(-2)
    # print(f"world points bounds: {pts_world.reshape(-1,3).min(dim=0)[0]} to {pts_world.reshape(-1,3).max(dim=0)[0]}")
    rays_time = time.time()

    # move source points to target screen space
    tgt_pts_screen = tgt_frame.camera.transform_points_screen(pts_world.squeeze(), image_size=(imh, imw))

    # move source camera center to target screen space
    src_center_tgt_screen = tgt_frame.camera.transform_points_screen(src_frame.camera.get_camera_center(), image_size=(imh, imw)).squeeze()

    # build epipolar mask (draw lines from source camera center to source points in target screen space)
    # start: source camera center, end: source points in target screen space

    # get flow of points 
    center_to_pts_flow = tgt_pts_screen[...,:2] - src_center_tgt_screen[...,:2]

    # normalize flow
    center_to_pts_flow = center_to_pts_flow / center_to_pts_flow.norm(dim=-1, keepdim=True)

    # get slope and intercept of lines
    slope = center_to_pts_flow[:,:,0:1] / center_to_pts_flow[:,:,1:2]
    intercept = tgt_pts_screen[:,:, 0:1] - slope * tgt_pts_screen[:,:, 1:2]

    # find intersection of lines with tgt screen (x = 0, x = imw, y = 0, y = imh)
    left = slope * 0 + intercept
    left_sane = (left <= imh) & (0 <= left)
    left = torch.cat([left, torch.zeros_like(left)], dim=-1)

    right = slope * imw + intercept
    right_sane = (right <= imh) & (0 <= right)
    right = torch.cat([right, torch.ones_like(right) * imw], dim=-1)

    top = (0 - intercept) / slope
    top_sane = (top <= imw) & (0 <= top)
    top = torch.cat([torch.zeros_like(top), top], dim=-1)

    bottom = (imh - intercept) / slope
    bottom_sane = (bottom <= imw) & (0 <= bottom)
    bottom = torch.cat([torch.ones_like(bottom) * imh, bottom], dim=-1)

    # find intersection of lines
    points_one = torch.zeros_like(left)
    points_two = torch.zeros_like(left)

    # collect points from [left, right, bottom, top] in sequence
    points_one = torch.where(left_sane.repeat(1,1,2), left, points_one)

    points_one_zero = (points_one.sum(dim=-1) == 0).unsqueeze(-1).repeat(1,1,2)
    points_one = torch.where(right_sane.repeat(1,1,2) & points_one_zero, right, points_one)

    points_one_zero = (points_one.sum(dim=-1) == 0).unsqueeze(-1).repeat(1,1,2)
    points_one = torch.where(bottom_sane.repeat(1,1,2) & points_one_zero, bottom, points_one)

    points_one_zero = (points_one.sum(dim=-1) == 0).unsqueeze(-1).repeat(1,1,2)
    points_one = torch.where(top_sane.repeat(1,1,2) & points_one_zero, top, points_one)

    # collect points from [top, bottom, right, left] in sequence (opposite)
    points_two = torch.where(top_sane.repeat(1,1,2), top, points_two)

    points_two_zero = (points_two.sum(dim=-1) == 0).unsqueeze(-1).repeat(1,1,2)
    points_two = torch.where(bottom_sane.repeat(1,1,2) & points_two_zero, bottom, points_two)

    points_two_zero = (points_two.sum(dim=-1) == 0).unsqueeze(-1).repeat(1,1,2)
    points_two = torch.where(right_sane.repeat(1,1,2) & points_two_zero, right, points_two)

    points_two_zero = (points_two.sum(dim=-1) == 0).unsqueeze(-1).repeat(1,1,2)
    points_two = torch.where(left_sane.repeat(1,1,2) & points_two_zero, left, points_two)

    # if source point lies inside target screen (find only one intersection)
    if (imh >= src_center_tgt_screen[0] >= 0) and (imw >= src_center_tgt_screen[1] >= 0):
        points_one_flow = points_one - src_center_tgt_screen[:2]
        points_one_flow_direction = (points_one_flow > 0)

        points_two_flow = points_two - src_center_tgt_screen[:2]
        points_two_flow_direction = (points_two_flow > 0)

        orig_flow_direction = (center_to_pts_flow > 0)

        # if flow direction is same as orig flow direction, pick points_one, else points_two
        points_one_alinged = (points_one_flow_direction == orig_flow_direction).all(dim=-1).unsqueeze(-1).repeat(1,1,2)
        points_one = torch.where(points_one_alinged, points_one, points_two)

        # points two is source camera center
        points_two = points_two * 0 + src_center_tgt_screen[:2]
    
    # if debug terminate with depth 
    if debug_depth:
        # remove points that are out of bounds (in target screen space)
        tgt_pts_screen_mask = (tgt_pts_screen[...,:2] < 0) | (tgt_pts_screen[...,:2] > imh)
        tgt_pts_screen_mask = ~tgt_pts_screen_mask.any(dim=-1, keepdim=True)

        depth_dist = torch.norm(src_center_tgt_screen[:2] - tgt_pts_screen[...,:2], dim=-1, keepdim=True)
        points_one_dist = torch.norm(src_center_tgt_screen[:2] - points_one, dim=-1, keepdim=True)
        points_two_dist = torch.norm(src_center_tgt_screen[:2] - points_two, dim=-1, keepdim=True)

        # replace where reprojected point is closer to source camera on target screen
        points_one = torch.where((depth_dist < points_one_dist) & tgt_pts_screen_mask, tgt_pts_screen[...,:2], points_one)
        points_two = torch.where((depth_dist < points_two_dist) & tgt_pts_screen_mask, tgt_pts_screen[...,:2], points_two)

    # build epipolar mask
    attention_mask = torch.zeros((imh * imw, imh, imw), dtype=torch.bool, device=src_frame.camera.device)

    # quantize points to pixel indices
    points_one = (points_one - 0.5).reshape(-1,2).long().numpy()
    points_two = (points_two - 0.5).reshape(-1,2).long().numpy()
    
    # cache only supports 32x32 epipolar mask with 3x3 dilation
    if not (imh == 32 and imw == 32) or not dialate_mask or masks32 is None:
        # iterate over points_one and points_two together and draw lines
        for idx, (p1, p2) in enumerate(zip(points_one, points_two)):
            # skip out of bounds points
            if p1.sum() == 0 and p2.sum() == 0:
                continue
            
            if not dialate_mask:
                # draw line from p1 to p2
                rr, cc = line(int(p1[1]), int(p1[0]), int(p2[1]), int(p2[0]), use_cache=False)
                rr, cc = rr.astype(np.int32), cc.astype(np.int32)
                attention_mask[idx, rr, cc] = True
            else:
                # draw lines with mask dilation (from all neighbors of p1 to neighbors of p2)
                rrs, ccs = [], []
                for dx, dy in [(0,0), (0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1)]:  # 8 neighbors
                    _p1 = [min(max(p1[0] + dy, 0), imh - 1), min(max(p1[1] + dx, 0), imw - 1)]
                    _p2 = [min(max(p2[0] + dy, 0), imh - 1), min(max(p2[1] + dx, 0), imw - 1)]
                    rr, cc = line(int(_p1[1]), int(_p1[0]), int(_p2[1]), int(_p2[0]))
                    rrs.append(rr); ccs.append(cc)
                rrs, ccs = np.concatenate(rrs), np.concatenate(ccs)
                attention_mask[idx, rrs.astype(np.int32), ccs.astype(np.int32)] = True
    else:
        points_one_y, points_one_x = points_one[:,0], points_one[:,1]
        points_two_y, points_two_x = points_two[:,0], points_two[:,1]
        attention_mask = masks32[points_one_y, points_one_x, points_two_y, points_two_x]
        attention_mask = torch.from_numpy(attention_mask).to(src_frame.camera.device)

    # reshape to (imh, imw, imh, imw)
    attention_mask = attention_mask.reshape(imh * imw, imh * imw)

    # stores flattened 2D attention mask 
    if visualize_mask:
        attention_mask = attention_mask.reshape(imh * imw, imh * imw)
        am_img = (attention_mask.squeeze().unsqueeze(-1).repeat(1,1,3).float().numpy() * 255).astype(np.uint8)
        imageio.imsave("data/visuals/epipolar_masks/batched_mask.png", am_img)

    return attention_mask


def get_opencv_from_blender(matrix_world, fov, image_size):
    # convert matrix_world to opencv format extrinsics
    opencv_world_to_cam = matrix_world.inverse()
    opencv_world_to_cam[1, :] *= -1
    opencv_world_to_cam[2, :] *= -1
    R, T = opencv_world_to_cam[:3, :3], opencv_world_to_cam[:3, 3]
    R, T = R.unsqueeze(0), T.unsqueeze(0)
    
    # convert fov to opencv format intrinsics
    focal = 1 / np.tan(fov / 2)
    intrinsics = np.diag(np.array([focal, focal, 1])).astype(np.float32)
    opencv_cam_matrix = torch.from_numpy(intrinsics).unsqueeze(0).float()
    opencv_cam_matrix[:, :2, -1] += torch.tensor([image_size / 2, image_size / 2])
    opencv_cam_matrix[:, [0,1], [0,1]] *= image_size / 2

    return R, T, opencv_cam_matrix


def compute_plucker_embed(frame, imw, imh):
    """ Computes Plucker coordinates for a Pytorch3D camera. """

    # get camera center
    cam_pos = frame.camera.get_camera_center()

    # get ray bundle
    src_ray_bundle = NDCMultinomialRaysampler(
        image_width=imw,
        image_height=imh,
        n_pts_per_ray=1,
        min_depth=1.0,
        max_depth=1.0,
    )(frame.camera)
    
    # get ray directions
    ray_dirs = F.normalize(src_ray_bundle.directions, dim=-1)

    # get plucker coordinates
    cross = torch.cross(cam_pos[:,None,None,:], ray_dirs, dim=-1)
    plucker = torch.cat((ray_dirs, cross), dim=-1)
    plucker = plucker.permute(0, 3, 1, 2)

    return plucker  # (B, 6, H, W, )


def cartesian_to_spherical(xyz):
    xy = xyz[:,0]**2 + xyz[:,1]**2
    z = np.sqrt(xy + xyz[:,2]**2)
    theta = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from z-axis down
    azimuth = np.arctan2(xyz[:,1], xyz[:,0])
    return np.stack([theta, azimuth, z], axis=-1)


def spherical_to_cartesian(spherical_coords):
    # convert from spherical to cartesian coordinates
    theta, azimuth, radius = spherical_coords.T
    x = radius * np.sin(theta) * np.cos(azimuth)
    y = radius * np.sin(theta) * np.sin(azimuth)
    z = radius * np.cos(theta)
    return np.stack([x, y, z], axis=-1)


def look_at(eye, center, up):
    # Create a normalized direction vector from eye to center
    f = np.array(center) - np.array(eye)
    f /= np.linalg.norm(f)

    # Create a normalized right vector
    up_norm = np.array(up) / np.linalg.norm(up)
    s = np.cross(f, up_norm)
    s /= np.linalg.norm(s)

    # Recompute the up vector
    u = np.cross(s, f)

    # Create rotation matrix R
    R = np.array([[s[0], s[1], s[2]],
                  [u[0], u[1], u[2]],
                  [-f[0], -f[1], -f[2]]])

    # Create translation vector T
    T = -np.dot(R, np.array(eye))

    return R, T


def get_blender_from_spherical(elevation, azimuth):
    """ Generates blender camera from spherical coordinates. """

    cartesian_coords = spherical_to_cartesian(np.array([[elevation, azimuth, 3.5]]))
    
    # get camera rotation
    center = np.array([0, 0, 0])
    eye = cartesian_coords[0]
    up = np.array([0, 0, 1])

    R, T = look_at(eye, center, up)
    R = R.T; T = -np.dot(R, T)
    RT = np.concatenate([R, T.reshape(3,1)], axis=-1)

    blender_cam = torch.from_numpy(RT).float()
    blender_cam = torch.cat([blender_cam, torch.tensor([[0, 0, 0, 1]])], axis=0)
    return blender_cam


def get_mask_and_plucker(src_frame, tgt_frame, image_size, dialate_mask=True, debug_depth=False, visualize_mask=False):
    """ Given a pair of source and target frames (blender outputs), returns the epipolar attention masks and plucker embeddings."""

    # get pytorch3d frames (blender to opencv, then opencv to pytorch3d)
    src_R, src_T, src_intrinsics = get_opencv_from_blender(src_frame["camera"], src_frame["fov"], image_size)
    src_camera_pytorch3d = cameras_from_opencv_projection(src_R, src_T, src_intrinsics, torch.tensor([image_size, image_size]).float().unsqueeze(0))
    src_frame.update({"camera": src_camera_pytorch3d})

    tgt_R, tgt_T, tgt_intrinsics = get_opencv_from_blender(tgt_frame["camera"], tgt_frame["fov"], image_size)
    tgt_camera_pytorch3d = cameras_from_opencv_projection(tgt_R, tgt_T, tgt_intrinsics, torch.tensor([image_size, image_size]).float().unsqueeze(0))
    tgt_frame.update({"camera": tgt_camera_pytorch3d})

    # compute epipolar masks
    image_height, image_width = image_size, image_size
    src_mask = compute_epipolar_mask(src_frame, tgt_frame, image_height, image_width, dialate_mask, debug_depth, visualize_mask)
    tgt_mask = compute_epipolar_mask(tgt_frame, src_frame, image_height, image_width, dialate_mask, debug_depth, visualize_mask)

    # compute plucker coordinates
    src_plucker = compute_plucker_embed(src_frame, image_height, image_width).squeeze()
    tgt_plucker = compute_plucker_embed(tgt_frame, image_height, image_width).squeeze()

    return src_mask, tgt_mask, src_plucker, tgt_plucker


def get_batch_from_spherical(elevations, azimuths, fov=0.702769935131073, image_size=256):
    """Given a list of elevations and azimuths, generates cameras, computes epipolar masks and plucker embeddings and organizes them as a batch."""

    num_views = len(elevations)
    latent_size = image_size // 8
    assert len(elevations) == len(azimuths)

    # intialize all epipolar masks to ones (i.e. all pixels are considered)
    batch_attention_masks = torch.ones(num_views, num_views, latent_size ** 2, latent_size ** 2, dtype=torch.bool)
    plucker_embeds = [None for _ in range(num_views)]

    # compute pairwise mask and plucker
    for i, icam in enumerate(zip(elevations, azimuths)):
        for j, jcam in enumerate(zip(elevations, azimuths)):
            if i == j: continue

            first_frame = edict({"fov": fov}); second_frame = edict({"fov": fov})
            first_frame["camera"] = get_blender_from_spherical(elevation=icam[0], azimuth=icam[1])
            second_frame["camera"] = get_blender_from_spherical(elevation=jcam[0], azimuth=jcam[1])
            first_mask, second_mask, first_plucker, second_plucker = get_mask_and_plucker(first_frame, second_frame, latent_size, dialate_mask=True)

            batch_attention_masks[i, j], batch_attention_masks[j, i] = first_mask, second_mask
            plucker_embeds[i], plucker_embeds[j] = first_plucker, second_plucker

    # organize as batch
    batch = {}
    batch_attention_masks = rearrange(batch_attention_masks, 'b1 b2 h w -> (b1 h) (b2 w)')
    batch["epi_constraint_masks"] = batch_attention_masks
    batch["plucker_embeds"] = torch.stack(plucker_embeds)

    return batch
