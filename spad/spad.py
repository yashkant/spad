import os
import numpy as np
import torch
import torch.nn as nn

from einops import rearrange, repeat
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.modules.ema import LitEma
from ldm.modules.diffusionmodules.util import zero_module
from itertools import chain


class SPAD(LatentDiffusion):
    """Spatially Aware Multiview Diffusers class."""

    def __init__(self, *args, **kwargs):
        self.cc_type = "timesteps_only_emb"
        self.cfg_conds = kwargs.pop("cfg_conds", ["txt"])  # classifier-free guidance
        self.cfg_scales = kwargs.pop("cfg_scales", [7.5])  # classifier-free guidance
        self.use_abs_extrinsics = kwargs.pop("use_abs_extrinsics", False)  # use extrinsic for conditioning
        self.use_intrinsic = kwargs.pop("use_intrinsic", False)  # use intrinsic for conditioning

        # assert order and size
        assert self.cfg_conds == ['txt', 'cam', 'epi', 'plucker'][:len(self.cfg_conds)]
        assert len(self.cfg_conds) == len(self.cfg_scales)

        # init superclass
        LatentDiffusion.__init__(self, *args, **kwargs)

        # timesteps conditioning
        self.cc_projection = nn.Sequential(
            nn.Linear(4, 1280) if not self.use_intrinsic else nn.Linear(8, 1280),
            nn.SiLU(),
            zero_module(nn.Linear(1280, 1280)),
        )
        
        # assert flags
        self.range_assert = -1

    def reinit_ema(self):
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

    @torch.no_grad()
    def get_input(self, batch, return_first_stage_outputs=False, force_c_encode=False, return_original_cond=False, bs=None, return_uc=False):
        # img (z): [n_objects, n_views, 4, 64, 64]
        # txt (c): [n_objects, n_views, max_seq_len, 768]
        # cam (T): [n_objects, n_views, 1280]
        
        use_laion_batch = False

        n_objects, n_views = batch["img"].shape[:2]
        if self.use_abs_extrinsics:
            # absolute camera pose
            cam_extrinsics = rearrange(torch.diagonal(batch["cam"], dim1=1, dim2=2), "n c v -> n v c")
        else:
            # relative camera pose (first view)
            cam_extrinsics = batch["cam"][:, 0] 

        batch["cam"] = cam_extrinsics if not self.use_intrinsic else torch.cat([cam_extrinsics, batch["render_intrinsics_flat"]], dim=-1)

        # hardcoded keys
        input_type = batch["img"].dtype
        x = rearrange(batch["img"], "n v h w c -> n v c h w").to(memory_format=torch.contiguous_format).to(self.device).to(input_type)
        xtxt = np.array(batch["txt"]).T.tolist()  # transpose to [n_views, n_objects]
        cam = batch['cam'].to(memory_format=torch.contiguous_format).to(self.device).to(input_type)
        
        # assert shapes
        assert x.shape[0] == cam.shape[0] == len(xtxt)
        assert x.shape[1] == cam.shape[1] == len(xtxt[0])
        self.get_input_asserts([x, cam, xtxt])

        # crop to batch size
        if bs is not None:
            x, cam, xtxt = x[:bs], cam[:bs], xtxt[:bs]
            if "epi_constraint_masks" in batch and not skip_epi:
                batch["epi_constraint_masks"] = batch["epi_constraint_masks"][:bs]
            if "plucker_embeds" in batch and not skip_plucker:
                batch["plucker_embeds"] = batch["plucker_embeds"][:bs]

        # rearrange to [n_objects * n_views, ...]
        n_objects, n_views = x.shape[:2]
        x = rearrange(x, "n v c h w -> (n v) c h w")
        xtxt = list(chain(*xtxt)) # list of lists to list
        cam = rearrange(cam, "n v c -> (n v) c")

        # encode images
        z = self.get_first_stage_encoding(self.encode_first_stage(x))
        z = z.detach()
        z = rearrange(z, "(n v) c h w -> n v c h w", v=n_views)

        cond = {}
        c,h,w = z.shape[2:]

        # text conditioning
        clip_emb = self.get_learned_conditioning(xtxt)
        clip_emb = clip_emb.detach()
        clip_emb = rearrange(clip_emb, "(n v) s c -> n v s c", v=n_views)
        
        # generate camera embeddings
        with torch.enable_grad():
            cam_emb = self.cc_projection(cam).squeeze(1)
            cam_emb = rearrange(cam_emb, "(n v) c -> n v c", v=n_views)

        # expand views back to original shape
        x = rearrange(x, "(n v) c h w -> n v c h w", v=n_views)

        # add as crossattn condition
        cond["c_crossattn"] = [clip_emb, cam_emb]
        cond["c_concat"] = []

        # add epipolar masks and plucker embeddings
        cond["c_crossattn"].append(batch["epi_constraint_masks"])
        cond["c_concat"].append(batch["plucker_embeds"])

        # return outputs
        out = [z, cond]

        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])

        if return_original_cond:
            out.append(batch.get("txt", [""] * len(x)))
        
        # create unconditional conditioning (inference)
        if return_uc:
            if "txt" in self.cfg_conds:
                uc_clip_emb = self.get_learned_conditioning([""]).detach()
                uc_clip_emb = repeat(uc_clip_emb, "1 c d -> n v c d", n=n_objects, v=n_views)
            else:
                raise ValueError(f"txt not in cfg_conds: {self.cfg_conds}")
            uc_cam_emb = cam_emb
            ucond = {
                "c_crossattn": [uc_clip_emb, uc_cam_emb, batch["epi_constraint_masks"]],
                "c_concat": [c_emb.clone().detach() for c_emb in cond["c_concat"]]
            }
            out.append(ucond)

        return out

    def get_input_asserts(self, inputs):
        # asserts range and conditions for early iterations
        x, cam, xtxt = inputs
        if self.range_assert > 0:
            print(f"================= range assertion =================")
            assert torch.all(x >= -1.0) and torch.all(x <= 1.0), f"image range assertion failed: {x.min()}, {x.max()}"
            print(f"cam: {cam[:4]}") 
            print(f"xtxt: {xtxt[:4]}")
            print(f"cam shape: {cam.shape} and xtxt shape: {len(xtxt)}")
            self.range_assert -= 1
    
    def decode_first_stage(self, z, *args, **kwargs):
        if len(z.shape) == 5:
            num_objects, num_views = z.shape[:2]
            z = rearrange(z, "n v c h w -> (n v) c h w")
            z = LatentDiffusion.decode_first_stage(self, z, *args, **kwargs)
            z = rearrange(z, "(n v) c h w -> n v c h w", v=num_views)
        else:
            z = LatentDiffusion.decode_first_stage(self, z, *args, **kwargs)
        return z

    def encode_first_stage(self, *args, **kwargs):
        return LatentDiffusion.encode_first_stage(self, *args, **kwargs)

