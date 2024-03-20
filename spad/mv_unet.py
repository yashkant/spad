from abc import abstractmethod
from functools import partial
import math
from typing import Iterable

import numpy as np
import torch

from ldm.modules.diffusionmodules.util import timestep_embedding
from spad.mv_attention import SPADTransformer as SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepBlock
from einops import rearrange


class SPADUnetModel(UNetModel):
    """ Modified UnetModel to support simlutaneous denoising of many views. """

    def __init__(self, *args, **kwargs):
        UNetModel.__init__(self, *args, **kwargs)

    def encode(self, h, emb, context, blocks):
        hs = []
        n_objects, n_views = h.shape[:2]
        for i, block in enumerate(blocks):
            for j, layer in enumerate(block):
                if isinstance(layer, SpatialTransformer):
                    h = layer(h, context)
                elif isinstance(layer, TimestepBlock):
                    # squash first two dims (single pass)
                    h = rearrange(h, "n v c h w -> (n v) c h w")
                    emb = rearrange(emb, "n v c -> (n v) c")
                    # apply layer
                    h = layer(h, emb)
                    # unsquash first two dims
                    h = rearrange(h, "(n v) c h w -> n v c h w", n=n_objects, v=n_views)
                    emb = rearrange(emb, "(n v) c -> n v c", n=n_objects, v=n_views)
                else:
                    # squash first two dims (single pass)
                    h = rearrange(h, "n v c h w -> (n v) c h w")
                    # apply layer
                    h = layer(h)
                    # unsquash first two dims
                    h = rearrange(h, "(n v) c h w -> n v c h w", n=n_objects, v=n_views)
                
                if h.isnan().any():
                    breakpoint()

            hs.append(h)
        return hs

    def decode(self, h, hs, emb, context, xdtype, last=False, return_outputs=False):
        ho = []
        n_objects, n_views = h.shape[:2]
        for i, block in enumerate(self.output_blocks):
            h = torch.cat([h, hs[-(i+1)]], dim=2)
            for j, layer in enumerate(block):
                if isinstance(layer, SpatialTransformer):
                    h = layer(h, context)

                elif isinstance(layer, TimestepBlock):
                    # squash first two dims (single pass)
                    h = rearrange(h, "n v c h w -> (n v) c h w")
                    emb = rearrange(emb, "n v c -> (n v) c")
                    # apply layer
                    h = layer(h, emb)
                    # unsquash first two dims
                    h = rearrange(h, "(n v) c h w -> n v c h w", n=n_objects, v=n_views)
                    emb = rearrange(emb, "(n v) c -> n v c", n=n_objects, v=n_views)
                else:
                    # squash first two dims (single pass)
                    h = rearrange(h, "n v c h w -> (n v) c h w")
                    # apply layer
                    h = layer(h)
                    # unsquash first two dims
                    h = rearrange(h, "(n v) c h w -> n v c h w", n=n_objects, v=n_views)
            ho.append(h)

        # process last layer
        h = h.type(xdtype)
        h = rearrange(h, "n v c h w -> (n v) c h w")
        if last:
            if self.predict_codebook_ids:
                # not used in vae
                h = self.id_predictor(h)
            else:
                h = self.out(h)
        h = rearrange(h, "(n v) c h w -> n v c h w", n=n_objects, v=n_views)

        ho.append(h)
        return ho if return_outputs else h

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        """
        objaverse batch:
        # img (x): [n_objects, n_views, 4, 64, 64]
        # timesteps (timesteps): [n_objects, n_views]
        # txt (context[0]): [n_objects, n_views, max_seq_len, 768]
        # cam (context[1]): [n_objects, n_views, 1280]

        laion batch:
        # img (x): [batch_size, 1, 4, 64, 64]
        # timesteps (timesteps): [batch_size, 1, 1]
        # txt (context[0]): [batch_size, 1, max_seq_len, 768]
        # cam (context[1]): [batch_size, 1, 1280] * 0.0

        :return: an [n_objects, n_views, 4, 64, 64]
        """
        n_objects, n_views = x.shape[:2]

        # timsteps embedding
        timesteps = rearrange(timesteps, "n v -> (n v)")
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        time = self.time_embed(t_emb)
        time = rearrange(time, "(n v) d -> n v d", n=n_objects, v=n_views)

        # extract txt and cam embedding (absolute) from context
        if len(context) == 2:
            txt, cam = context
        elif len(context) == 3:
            txt, cam, epi_mask = context
            txt = (txt, epi_mask)
        else:
            raise ValueError
        
        # extract plucker embedding from x
        if x.shape[2] > 4:
            plucker, x = x[:, :, 4:], x[:, :, :4]
            txt = (*txt, plucker) if isinstance(txt, tuple) else (txt, plucker)

        # combine timestep and camera embedding (resnet)
        time_cam = time + cam
        del time, cam

        # encode
        h = x.type(self.dtype)
        hs = self.encode(h, time_cam, txt, self.input_blocks)

        # middle block
        h = self.encode(hs[-1], time_cam, txt, [self.middle_block])[0]
        
        # decode
        h = self.decode(h, hs, time_cam, txt, x.dtype, last=True)

        # concat along channel dim
        return h


if __name__ == "__main__":
    model_args = {
        "image_size": 32, # unused
        "in_channels": 4,
        "out_channels": 4,
        "model_channels": 320,
        "attention_resolutions": [ 4, 2, 1 ],
        "num_res_blocks": 2,
        "channel_mult": [ 1, 2, 4, 4 ],
        "num_heads": 8,
        "use_spatial_transformer": True,
        "transformer_depth": 1,
        "context_dim": 768,
        "use_checkpoint": False,
        "legacy": False,
    }

    # manyviews unet
    model = SPADUnetModel(**model_args)
    model.cuda()
    model.post_init()
    model.eval()
    n_objects = 2; n_views = 3

    # img (z): [n_objects, n_views, 4, 64, 64]
    # txt (c): [n_objects, n_views, max_seq_len, 768]
    # cam (T): [n_objects, n_views, 1280]

    x = torch.randn(n_objects, n_views, 10, 32, 32).cuda()
    timesteps = torch.randint(0, 1000, (n_objects, n_views, )).long().cuda()
    context = [
        torch.randn(n_objects, n_views, 77, 768).cuda(), 
        torch.randn(n_objects, n_views, 1280).cuda(),
        torch.ones(n_objects, n_views * 32 * 32, n_views * 32 * 32, dtype=torch.bool).cuda()
    ]
    context[-1][0] = False
    out = model(x, timesteps=timesteps, context=context)
    print(f"in: {x.shape}, out: {out.shape}")
    breakpoint()

