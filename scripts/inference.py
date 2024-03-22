import os
import torch
import numpy as np
import math
import imageio
import time
import argparse

from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from einops import rearrange
from itertools import chain, cycle
from tqdm import tqdm

from spad.utils import load_model_from_config, slugify
from spad.geometry import get_batch_from_spherical

from ldm.models.diffusion.ddim import ManyViewDDIMSampler


def generate_batch(elevations=[60,60,60,60], azimuths=[0,90,180,270], use_abs=False):
    elevations = [math.radians(e) for e in elevations]; azimuths = [math.radians(a) for a in azimuths]

    # assert first frame is identity
    batch = get_batch_from_spherical(elevations, azimuths)

    # generate 12 camera poses at three elevations (+/- 30) and four azimuths (0, 90, 180, 270)
    abs_cams = []
    for theta,azimuth in zip(elevations,azimuths):
        abs_cams.append(torch.tensor([theta, azimuth, 3.5]))

    # generate relative ones
    debug_cams = [[] for _ in range(len(azimuths))]
    for i, icam in enumerate(abs_cams):
        for j, jcam in enumerate(abs_cams):
            if use_abs:
                dcam = torch.tensor([icam[0], math.sin(icam[1]), math.cos(icam[1]), icam[2]])
            else:
                dcam = icam - jcam
                dcam = torch.tensor([dcam[0].item(), math.sin(dcam[1].item()), math.cos(dcam[1].item()), dcam[2].item()])
            debug_cams[i].append(dcam)
    
    batch["cam"] = torch.stack([torch.stack(dc) for dc in debug_cams])

    # put intrinsics in the batch
    focal = 1 / np.tan(0.702769935131073 / 2)
    intrinsics = np.diag(np.array([focal, focal, 1])).astype(np.float32)
    intrinsics = torch.from_numpy(intrinsics).unsqueeze(0).float().repeat(batch["cam"].shape[0], 1, 1)
    batch["render_intrinsics_flat"] = intrinsics[ :, [0,1,0,1], [0,1,-1,-1]] 

    return batch


def get_gaussian_image(blob_width=256, blob_height=256, sigma=0.5):
    # create a black 2D gaussian blob with white background
    X = np.linspace(-1, 1, blob_width)[None, :]
    Y = np.linspace(-1, 1, blob_height)[:, None]
    inv_dev = 1 / sigma ** 2
    gaussian_blob = np.exp(-0.5 * (X**2) * inv_dev) * np.exp(-0.5 * (Y**2) * inv_dev)
    if gaussian_blob.max() > 0:
        gaussian_blob =  255.0 * (gaussian_blob - gaussian_blob.min()) / gaussian_blob.max()
    gaussian_blob = 255.0 - gaussian_blob
    
    # normalize -1,1 and return 3 channel
    gaussian_blob = (gaussian_blob / 255.0) * 2.0 - 1.0
    gaussian_blob = np.expand_dims(gaussian_blob, axis=-1).repeat(3,-1)
    gaussian_blob = torch.from_numpy(gaussian_blob)

    # return gaussian_blob.astype(np.uint8)
    return gaussian_blob


def denoise(batch, model, device, idx, total_views, outpath, blob_sigma, ddim_steps):
    # get paired sampler
    ddim_sampler = ManyViewDDIMSampler(model)
    
    # get input from model, and prepare for sampling
    log = dict()
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}

    # gaussian blob init (helps prevent non-white background issues)
    blob = get_gaussian_image(sigma=blob_sigma)
    batch["img"][:,:] = blob
    print(f"using gaussian initialization")

    # get model inputs    
    z, c, x, xrec, xc, uc = model.get_input(batch, return_first_stage_outputs=True, force_c_encode=True, return_original_cond=True, return_uc=True)

    # sample with unconditional conditioning
    shape = (model.channels, model.image_size, model.image_size)
    batch_size = (len(x), total_views) # (batch_size, num_views)
    
    # run paired sampler
    kwargs = dict(
        unconditional_conditioning=uc,
        x0=z, # blob init
    )
    samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, c, verbose=False, **kwargs)
    x_samples_cfg = model.decode_first_stage(samples)
    x_samples_cfg = torch.clamp(x_samples_cfg, -1., 1.)

    # flatten images and captions
    x_samples = rearrange(x_samples_cfg, "b v c h w -> (b v) c h w")
    x_samples = ((x_samples + 1.0) / 2.0) 
    xtxt = np.array(batch["txt"]).T.tolist()
    xtxt = list(chain(*xtxt)) # list of lists to list
    x_samples = rearrange(x_samples, "(n v) c h w -> n h (v w) c", v= total_views)
    x_samples = (x_samples * 255.0).cpu().float().numpy().astype(np.uint8)

    # store images
    os.makedirs(outpath, exist_ok=True)
    for _idx, (image, caption) in enumerate(zip(x_samples, xtxt)):
        caption = slugify(caption)
        save_path = f"{outpath}/{caption}.png"
        imageio.imsave(save_path, image)
        print(f"saved image: {save_path}")
    
    return log


def load_captions(path="data/1k_captions_viz.npy"):
    # load inference captions 
    captions = np.load(path, allow_pickle=True).tolist()
    
    # add [tdv] to captions (special token used in training)
    captions = ["[tdv] " + c if "[tdv]" not in c else c for c in captions ]
    return captions


def main(config_path, checkpoint_path, captions, cfg_scale=7.5, blob_sigma=0.5, batch_size=1, total_views=8, ddim_steps=100):
    seed_everything(42+69)

    # load model and sampler
    model_config = OmegaConf.load(config_path)
    model = load_model_from_config(model_config, checkpoint_path, verbose=True, inference_run=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device).eval()
    model.cfg_conds = ["txt"]
    model.cfg_scales = [cfg_scale]

    # setup logging and output directories
    visuals_dir = "data/visuals/"
    ts = str(round(time.time()))
    outdir = os.path.join(visuals_dir, "inference", ts) 
    os.makedirs(outdir, exist_ok=True)

    # add opt params
    dataloader = cycle([{
        "img": torch.zeros(batch_size, total_views, 256, 256, 3)
    }])

    # run inference
    terminate = False
    with torch.no_grad():
        # script runs in mixed precision, chosen automatically by apex
        for idx, batch in enumerate(tqdm(dataloader, desc="sampling")):
            
            # terminate when done
            if batch_size * (idx + 1) > len(captions):
                batch_size = len(captions) - batch_size * idx
                terminate = True

            # define cameras (elevations and azimuths)
            elevations = [60 for _ in range(total_views)] 
            azimuths = [az for az in np.linspace(0, 360* ((total_views-1)/total_views) , total_views)]

            print(f"using elevations: {elevations}, azimuths: {azimuths}")
            
            batch_cams = generate_batch(elevations, azimuths, use_abs=model.use_abs_extrinsics)
            batch_cams = {k: v[None].repeat_interleave(batch_size, dim=0).to(device) for k,v in batch_cams.items()}
            batch.update(batch_cams)

            # replace captions
            batch["txt"] = [captions[batch_size * idx: batch_size * (idx + 1)]] * total_views

            with model.ema_scope():
                # keep initial code fixed across all generated samples
                log = denoise(batch, model, device, idx, total_views, outdir, blob_sigma, ddim_steps)
            
            # terminate when done
            if terminate:
                break


if __name__ == "__main__":
    # available models
    model_zoo = {
        "spad_four_views": ("configs/spad_four_views.yaml", "data/checkpoints/spad_four_views.ckpt"),
        "spad_two_views": ("configs/spad_two_views.yaml", "data/checkpoints/spad_two_views.ckpt"),
        "spad_two_views_old": ("configs/spad_two_views_old.yaml", "/scratch/ssd004/scratch/yashkant/last.ckpt"),
    }

    # select model
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--captions", type=str, help="caption to use for inference, leave blank for using default captions.", default=None)
    argparser.add_argument("--model", type=str, default="spad_two_views")
    argparser.add_argument("--cfg_scale", type=float, default=7.5)
    argparser.add_argument("--blob_sigma", type=float, default=0.5)
    argparser.add_argument("--batch_size", type=int, default=1)
    argparser.add_argument("--total_views", type=int, default=8)
    argparser.add_argument("--ddim_steps", type=int, default=100)
    args = argparser.parse_args()

    # attach captions 
    if args.captions is not None:
        captions = eval(f'"{args.captions}"')
        captions = [captions] if isinstance(captions, str) else captions
        captions = ["[tdv] " + c if "[tdv]" not in c else c for c in captions ]
    else:
        captions = load_captions("data/captions_eval.npy")
    
    print(f"num of captions: {len(captions)}, batch_size: {args.batch_size}")

    # run inference
    config_path, checkpoint_path = model_zoo[args.model]
    main(config_path, checkpoint_path, captions, args.cfg_scale, args.blob_sigma, args.batch_size, args.total_views, args.ddim_steps)
