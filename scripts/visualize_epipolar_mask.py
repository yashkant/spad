import torch
import imageio
import os
import glob 
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm, trange
from easydict import EasyDict as edict
from spad.geometry import get_mask_and_plucker


def visualize_epipolar_mask(epipolar_attention_masks, object_frames, num_views, image_size):
    """
    Given epipolar attention masks as a grid (num_views x num_views x image_size^2 x image_size^2), visualizes epipolar lines (masks) between pairs of views.   
    """
    visuals_path = "data/visuals/epipolar_masks/"
    os.makedirs(visuals_path, exist_ok=True)

    # visualize epipolar mask (print epipolar line and pixel)
    step_size = 10
    for i in range(num_views):
        for j in range(num_views):
            if i == j:
                continue
            
            highlights = []
            for px in tqdm(range(0, image_size, step_size), total=(image_size//step_size), desc="drawing visuals"):
                for py in range(0, image_size, step_size):
                    
                    # epipolar mask
                    src_to_target_am = epipolar_attention_masks[i,j][px * image_size + py].reshape(image_size, image_size)

                    # highlight pixel in source view
                    src_image = object_frames[i].clone() 
                    src_image[:, px, py] *= 0
                    
                    # highlight epipolar mask in target view
                    tgt_image = object_frames[j].clone() 
                    tgt_image[:, src_to_target_am] *= 0

                    # concatenate images
                    image_concat = torch.cat([src_image, tgt_image], dim=2)
                    image_concat = (image_concat.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    highlights.append(image_concat)
            
            # save gif
            gif_path = os.path.join(visuals_path, f"epipolar_src_{i}_tgt_{j}.gif")
            imageio.mimsave(gif_path, highlights, fps=5)


def main(image_size=196, num_views=3, start_idx=1, object="lynx"):
    # randomly select an object and load few frames + cameras
    object_frames_paths = glob.glob(f"data/samples/lynx/images/*.png")
    object_frames_paths = sorted(object_frames_paths)[start_idx:start_idx+num_views]

    # load all frames
    object_frames = [torch.from_numpy(imageio.imread(frame_path)) for frame_path in object_frames_paths]

    # replace black with white background 
    object_frames = [torch.where(frame[:,:,-1:] < 255, torch.ones_like(frame) * 255.0, frame) for frame in object_frames]
    object_frames = [(frame[:,:,:3]).float() / 255.0 for frame in object_frames]

    # resize to 196 (adjust based on your RAM)
    size = image_size
    object_frames = [F.interpolate(frame[None].permute(0,3,1,2), size=(size, size)) for frame in object_frames]
    object_frames = [frame.squeeze() for frame in object_frames]

    # load all cameras
    object_cams_paths = object_frames_paths
    object_cameras = [np.load(frame_path.replace(".png", ".npy").replace("images", "cameras"), allow_pickle=True) for frame_path in object_cams_paths]
    
    # field-of-view (fov) and extrinsics (matrix_world)
    object_cameras_fov = [camera.item()['fov'] for camera in object_cameras]
    object_cameras = [torch.from_numpy(np.matrix(camera.item()['matrix_world']).reshape(4,4)).float() for camera in object_cameras]

    # load all depths
    object_depths = [torch.from_numpy(imageio.imread(frame_path.replace(".png", ".exr").replace("images", "depths"))) for frame_path in object_frames_paths]
    object_depths = [depth[:, :, :1] * 10.0 for depth in object_depths]
    object_depths = [F.interpolate(depth[None].permute(0,3,1,2), size=(size, size)).squeeze(0) for depth in object_depths]

    # placeholders for epipolar attention masks and plucker embeddings
    num_views = len(object_frames)
    image_size = max(object_frames[0].shape)
    epipolar_attention_masks = torch.zeros(num_views, num_views, image_size ** 2, image_size ** 2, dtype=torch.bool)
    plucker_embeds = [None for _ in range(num_views)]
    
    # select pairs of source and target frame
    # compute epipolar attention masks and plucker embeddings b/w each pair
    for src_idx in trange(len(object_frames), desc="computing pairwise epipolar masks"):
        for tgt_idx in range(src_idx + 1, len(object_frames)):
            src_image, tgt_image = object_frames[src_idx], object_frames[tgt_idx]
            src_camera, tgt_camera = object_cameras[src_idx], object_cameras[tgt_idx]
            src_depth, tgt_depth = object_depths[src_idx], object_depths[tgt_idx]
            src_fov, tgt_fov = object_cameras_fov[src_idx], object_cameras_fov[tgt_idx]

            src_frame = edict({
                "camera": src_camera,
                "image_rgb": src_image[None], # batch dimension
                "depth_map": src_depth[None], # batch dimension
                "fov": src_fov
            })

            tgt_frame = edict({
                "camera": tgt_camera,
                "image_rgb": tgt_image[None], # batch dimension
                "depth_map": tgt_depth[None], # batch dimension
                "fov": tgt_fov
            })

            # create attention mask and pluckers, and store them
            src_mask, tgt_mask, src_plucker, tgt_plucker = get_mask_and_plucker(src_frame, tgt_frame, image_size, dialate_mask=True, debug_depth=True, visualize_mask=False)
            
            epipolar_attention_masks[src_idx, tgt_idx] = src_mask
            epipolar_attention_masks[tgt_idx, src_idx] = tgt_mask
            plucker_embeds[src_idx], plucker_embeds[tgt_idx] = src_plucker, tgt_plucker

    # visualize epipolar mask
    visualize_epipolar_mask(epipolar_attention_masks, object_frames, num_views, image_size)


if __name__ == "__main__":
    # select objects from ["lynx", "bread", "bono"]
    # select num_views from 1 to 12
    # select start_idx from 1 to 12
    # select image_size from 32 to 512
    main(image_size=196, num_views=3, start_idx=1, object="lynx")