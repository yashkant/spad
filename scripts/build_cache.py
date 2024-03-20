import numpy as np
import torch
import imageio
import cv2
from tqdm import tqdm
import torch.nn.functional as Fu
from scipy.spatial import ConvexHull
from skimage.draw import line, polygon, polygon2mask


def cache_lines_dialated(width):
    """" builds and caches dialated (3x3) lines between all possible points in a 32x32 grid""""
    cache = np.zeros((width, width, width, width, width, width), dtype=np.bool)
    imh, imw = width, width
    size = width * width
    
    for start_x in tqdm(range(0, width)):
        for start_y in tqdm(range(0, width)):
            for end_x in range(0, width):
                for end_y in range(0, width):
                    p1 = (start_x, start_y)
                    p2 = (end_x, end_y)

                    # draw lines from all neighbors of p1 to neighbors of p2 (mask dilation)
                    rrs, ccs = [], []
                    for dx, dy in [
                        (0, 0),
                        (0, 1),
                        (1, 1),
                        (1, 0),
                        (1, -1),
                        (0, -1),
                        (-1, -1),
                        (-1, 0),
                        (-1, 1),
                    ]:  # 1+8 neighbors
                        _p1 = [
                            min(max(p1[0] + dy, 0), imh - 1),
                            min(max(p1[1] + dx, 0), imw - 1),
                        ]
                        _p2 = [
                            min(max(p2[0] + dy, 0), imh - 1),
                            min(max(p2[1] + dx, 0), imw - 1),
                        ]

                        # draw line using skimage
                        rr, cc = line(
                            int(_p1[1]), int(_p1[0]), int(_p2[1]), int(_p2[0])
                        )
                        rrs.append(rr)
                        ccs.append(cc)
                    rrs, ccs = np.concatenate(rrs), np.concatenate(ccs)

                    # store in cache
                    cache[
                        start_x,
                        start_y,
                        end_x,
                        end_y,
                        rrs.astype(np.int32),
                        ccs.astype(np.int32),
                    ] = True

    return cache


if __name__ == "__main__":
    width = 32
    cache = cache_lines_dialated(width)
    np.save(f"data/cache/masks{width}.npy", cache)
    
    # store cache
    print(f"cache size: {len(cache)} and saved to data/cache/masks{width}.npy")
