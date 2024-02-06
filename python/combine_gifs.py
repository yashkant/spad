import imageio
import os
import glob
import numpy as np


def stitch_gifs_same_size(gif_paths, save_path='combined.gif'):
    # Read all GIFs and store them in a list
    gifs = [imageio.mimread(path) for path in gif_paths]
    
    # Determine the number of frames, height, and width
    num_frames = len(gifs[0])
    height, width, _ = gifs[0][0].shape
    total_width = width * len(gif_paths)
    
    # Create a new GIF
    new_gif = []
    for frame_idx in range(num_frames):
        # Create a new frame with the appropriate shape
        new_frame = np.zeros((height, total_width, 3), dtype=np.uint8)
        
        # Stitch each frame side by side
        for idx, gif in enumerate(gifs):
            new_frame[:, idx*width:(idx+1)*width, :] = gif[frame_idx]
        
        new_gif.append(new_frame)
    
    # Save the new GIF
    imageio.mimsave(save_path, new_gif)
    
    return save_path

# Example usage
gif_paths = glob.glob(os.path.join('/Users/yash/Downloads/triplane-gifs/', '*.gif'))

gif_paths_half = gif_paths[:len(gif_paths)//2]
stitched_gif_path = stitch_gifs_same_size(gif_paths_half, save_path='combined1.gif')

gif_paths_half = gif_paths[len(gif_paths)//2:]
stitched_gif_path = stitch_gifs_same_size(gif_paths_half, save_path='combined2.gif')

print(stitched_gif_path)
