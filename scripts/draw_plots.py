import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
SCALE = 2.0


COLORS = [
    '#e41a1c',  # 'red',
    '#377eb8',  # 'blue',
    '#4daf4a',  # 'green',
    '#ff7f00',  # 'orange',
    '#984ea3',  # 'purple',
    '#dede00',  # 'yellow',
]

nvs_results = {
    'Objaverse': {
        'Zero-1-to-3': {'PSNR': 18.16, 'SSIM': 0.81, 'LPIPS': 0.201},
        'iNVS': {'PSNR': 20.52, 'SSIM': 0.81, 'LPIPS': 0.178},
        'SyncDreamer': {'PSNR': 19.51, 'SSIM': 0.84, 'LPIPS': 0.174},
        'SPAD': {'PSNR': 20.29, 'SSIM': 0.84, 'LPIPS': 0.166},
    },
    'GSO': {
        'Zero-1-to-3': {'PSNR': 16.10, 'SSIM': 0.82, 'LPIPS': 0.183},
        'iNVS': {'PSNR': 18.53, 'SSIM': 0.80, 'LPIPS': 0.180},
        'SyncDreamer': {'PSNR': 17.18, 'SSIM': 0.83, 'LPIPS': 0.178},
        'SPAD': {'PSNR': 17.99, 'SSIM': 0.83, 'LPIPS': 0.169},
    },
}


def plot_nvs_results(
    datasets,
    methods,
    metric='PSNR',
    fontsize=16 * SCALE,
    figsize=(6 * SCALE, 7 * SCALE),
    y_lim=None,
):
    # plot as bar chart
    # we plot only one metric, of all methods, on all datasets
    # each cluster of bars is a dataset
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(datasets))
    width = 0.8 / len(methods)
    all_y = []
    for i, method in enumerate(methods):
        y = [nvs_results[dataset][method][metric] for dataset in datasets]
        ax.bar(x + i * width, y, width * 0.9, label=method, color=COLORS[i], edgecolor='black',linewidth=2)
        all_y.append(y)
    ax.set_xticks(x + width * len(methods) / 2)
    ax.set_xticklabels(datasets, fontsize=fontsize)
    if metric == 'LPIPS':
        metric = r'LPIPS $\downarrow$'
    if metric == 'PSNR':
        metric = r'PSNR $\uparrow$'
    if metric == 'SSIM':
        metric = r'SSIM $\uparrow$'
    ax.set_ylabel(metric, fontsize=fontsize)
    ax.set_yticklabels([f'{y:.2f}' for y in ax.get_yticks()], fontsize=fontsize)
    # y_lim
    all_y = np.array(all_y)
    if y_lim is None:
        y_lim = [np.min(all_y) * 0.9, np.max(all_y) * 1.05]
    ax.set_ylim(y_lim)
    # put the legend as one row
    ax.legend(
        loc='upper center',
        ncol=2,
        fontsize=fontsize,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    # plt.show()
    save_dir = 'quant_results'
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/nvs_{metric.split(' ')[0]}.png")


@torch.no_grad()
def add_noise_to_img(img_fn):
    from moviepy.editor import ImageSequenceClip
    from diffusers import AutoencoderKL, DDPMScheduler
    from torchvision import transforms
    model_name = 'stabilityai/stable-diffusion-2-1-base'
    noise_scheduler = DDPMScheduler.from_pretrained(
        model_name,
        subfolder="scheduler",
        local_files_only=True
    )
    vae = AutoencoderKL.from_pretrained(
        model_name,
        subfolder="vae",
        local_files_only=True,
    ).cuda()
    img_preproc = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    # img is [H, 256*8, 3]
    # we first make it [256, 256*8, 3]
    ori_img = np.array(Image.open(img_fn).convert('RGB'))
    ori_h, ori_w, _ = ori_img.shape
    assert ori_h <= 256 and ori_w == 256 * 8
    img = np.ones((256, 256 * 8, 3), dtype=np.uint8) * 255
    img[:ori_h] = ori_img
    # make it [512, 512*8, 3]
    img = cv2.resize(img, (512*8, 512), interpolation=cv2.INTER_LINEAR)
    img = img_preproc(img)  # [3, 512, 512*8], [-1, 1]
    # make it [8, 3, 512, 512]
    img = torch.stack([img[:, :, i*512:(i+1)*512] for i in range(8)], dim=0)
    img = img.float().cuda()
    # encode into latents
    latents = vae.encode(img).latent_dist.sample() * 0.18215
    # add noise on timestep range(1000, 0, -20)
    all_noisy_imgs = []
    for t in tqdm(range(1000-1, 0, -20)):
        noise = torch.randn_like(latents[:1]).repeat_interleave(latents.shape[0], dim=0)
        timesteps = torch.ones(
            (latents.shape[0],), device=latents.device,
        ).long() * t
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        noisy_latents = 1. / 0.18215 * noisy_latents
        noisy_img = vae.decode(noisy_latents).sample
        noisy_img = (noisy_img / 2. + 0.5).clamp(0, 1)  # [8, 3, 512, 512], [0, 1]
        noisy_img = noisy_img.permute(0, 2, 3, 1).cpu().numpy()  # [8, 512, 512, 3]
        noisy_img = np.round(noisy_img * 255.).astype(np.uint8)
        noisy_img = np.concatenate([img for img in noisy_img], axis=1)  # [512, 512*8, 3]
        noisy_img = cv2.resize(noisy_img, (256*8, 256), interpolation=cv2.INTER_LINEAR)
        noisy_img = noisy_img[:ori_h]
        all_noisy_imgs.append(noisy_img)
    # make it a video, stay at the clean frame for 20 frames
    all_noisy_imgs += ([ori_img] * (len(all_noisy_imgs) // 2))
    clip = ImageSequenceClip(all_noisy_imgs, fps=10)
    clip.write_gif(f'{img_fn[:-4]}.gif', fps=10)
    # clip.write_videofile(f'{img_fn[:-4]}.mp4', fps=10)


def stack_noisy_img_w_caption():
    from moviepy.editor import ImageSequenceClip, VideoFileClip
    for i in range(6):
        img_fn = f'./final_teaser/{i}.png'
        gif_fn = f'./final_teaser/{i}.gif'
        cap_fn = f'./final_teaser/{i}_caption.png'
        # get the clean image
        img = np.array(Image.open(img_fn).convert('RGB'))
        gif = VideoFileClip(gif_fn)
        # get all the frames of gif
        gif_imgs = []
        for t in range(int(gif.duration * gif.fps)):
            gif_imgs.append(gif.get_frame(t / gif.fps))
        for _ in range(int(gif.duration * gif.fps)):
            gif_imgs.append(img)
        # get caption image
        cap = np.array(Image.open(cap_fn))
        cap = cap[8:-8]
        cap = np.stack([cap for _ in range(3)], axis=-1)
        cap = np.clip(cap.astype(np.int32) * 16, a_max=255, a_min=0).astype(np.uint8)
        cap = 255 - cap
        h, w, _ = cap.shape
        cap = cv2.resize(cap, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
        h, w, _ = cap.shape
        gif_w = gif.size[0]
        # put it at the center
        cap_img = np.ones((h, gif_w, 3), dtype=np.uint8) * 255
        dw = (gif_w - w) // 2
        cap_img[:, dw:dw+w] = cap
        # stack them together
        gif_imgs = [np.concatenate([img, cap_img], axis=0) for img in gif_imgs]
        # assert len(gif_imgs) == 76
        clip = ImageSequenceClip(gif_imgs, fps=10)
        clip.write_gif(f'{img_fn[:-4]}_w_caption.gif', fps=10)


def stack_teaser():
    from nerv.utils import vstack_video_clips
    video = vstack_video_clips(
        [f'./final_teaser/{i}_w_caption.gif' for i in range(6)], padding=0)
    video.write_gif('./final_teaser/teaser.gif')
    video.write_videofile('./final_teaser/teaser.mp4', fps=20)
    """
    from nerv.utils import VideoReader,save_video,read_video_clip
    vid = VideoReader('./final_teaser/teaser.mp4')
    frames = vid.read_video(True)
    save_video(frames[::2], './final_teaser/teaser-fps_20.mp4', fps=20, rgb2bgr=False)
    vid = read_video_clip('./final_teaser/teaser-fps_20.mp4')
    vid.write_videofile('./final_teaser/teaser-fps_20-mvpy.mp4', fps=20)
    """


if __name__ == '__main__':
    for metric in ['PSNR', 'SSIM', 'LPIPS']:
        plot_nvs_results(
            datasets=['Objaverse', 'GSO'],
            methods=['Zero-1-to-3', 'iNVS', 'SyncDreamer', 'SPAD'],
            metric=metric,
        )
