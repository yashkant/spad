from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from ldm.modules.diffusionmodules.util import checkpoint, conv_nd
from ldm.modules.attention import *

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False
print(f"XFORMERS_IS_AVAILBLE: {XFORMERS_IS_AVAILBLE}")


class SPADAttention(nn.Module):
    """Uses xformers to implement efficient epipolar masking for cross-attention between views."""

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask=None, views=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape

        # epipolar mask
        if mask is not None:
            mask = mask.unsqueeze(1)
            mask_shape = (q.shape[-2], k.shape[-2])

            # interpolate epipolar mask to match downsampled unet branch
            mask = (
                F.interpolate(mask.to(torch.uint8), size=mask_shape).bool().squeeze(1)
            )

            # repeat mask for each attention head
            mask = (
                mask.unsqueeze(1)
                .repeat(1, self.heads, 1, 1)
                .reshape(b * self.heads, *mask.shape[-2:])
            )

        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        with torch.autocast(enabled=False, device_type="cuda"):
            q, k, v = q.float(), k.float(), v.float()
            
            mask_inf = 1e9
            fmask = None
            if mask is not None:
                # convert to attention bias
                fmask = mask.float()
                fmask[fmask == 0] = -mask_inf
                fmask[fmask == 1] = 0
            
            # actually compute the attention, what we cannot get enough of
            out = xformers.ops.memory_efficient_attention(
                q, k, v, attn_bias=fmask, op=self.attention_op
            )

        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )

        # no nans
        if out.isnan().any():
            breakpoint()

        # cleanup
        del q, k, v
        return self.to_out(out)


class SPADTransformerBlock(nn.Module):
    """Modified SPAD transformer block that enables spatially aware cross-attention."""

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        disable_self_attn=False,
    ):
        super().__init__()
        attn_cls = SPADAttention
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None,
        )  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None, mask=None):
        return checkpoint(
            self.manystream_forward,
            (x, context, mask),
            self.parameters(),
            self.checkpoint,
        )

    def manystream_forward(self, x, context=None, mask=None):
        assert not self.disable_self_attn
        # x: [n, v, h*w, c]
        # context: [n, v, seq_len, d]
        n, v = x.shape[:2]

        # self-attention (between views) with 3d mask
        x = rearrange(x, "n v hw c -> n (v hw) c")
        x = self.attn1(self.norm1(x), context=None, mask=mask, views=v) + x
        x = rearrange(x, "n (v hw) c -> n v hw c", v=v)

        # cross-attention (to individual views)
        x = rearrange(x, "n v hw c -> (n v) hw c")
        context = rearrange(context, "n v seq d -> (n v) seq d")
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        x = rearrange(x, "(n v) hw c -> n v hw c", v=v)

        return x


class SPADTransformer(nn.Module):
    """Spatial Transformer block with post init to add cross attn."""

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,  # 2.1 vs 1.5 difference
        use_checkpoint=True,
    ):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0
            )
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                SPADTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    checkpoint=use_checkpoint,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

        # modify conv layers incorporate plucker coordinates
        self.post_init()

    def post_init(self):
        assert getattr(self, "post_intialized", False) is False, "already modified!"

        # inflate input conv block to attach plucker coordinates
        conv_block = self.proj_in
        conv_params = {
            k: getattr(conv_block, k)
            for k in [
                "in_channels",
                "out_channels",
                "kernel_size",
                "stride",
                "padding",
            ]
        }
        conv_params["in_channels"] += 6
        conv_params["dims"] = 2
        conv_params["device"] = conv_block.weight.device

        # copy original weights for input conv block
        inflated_proj_in = conv_nd(**conv_params)
        inp_weight = conv_block.weight.data
        feat_shape = inp_weight.shape

        # intialize new weights for plucker coordinates as zeros
        feat_weight = torch.zeros(
            (feat_shape[0], 6, *feat_shape[2:]), device=inp_weight.device
        )

        # assemble new weights and bias
        inflated_proj_in.weight.data.copy_(
            torch.cat([inp_weight, feat_weight], dim=1)
        )
        inflated_proj_in.bias.data.copy_(conv_block.bias.data)
        self.proj_in = inflated_proj_in
        self.post_intialized = True

    def forward(self, x, context=None):
        return self.spad_forward(x, context=context)

    def spad_forward(self, x, context=None):
        """
        x: tensor of shape [n, v, c (4), h (32), w (32)] 
        context: list of [text_emb, epipolar_mask, plucker_coords]
            - text_emb: tensor of shape [n, v, seq_len (77), dim (768)]
            - epipolar_mask: bool tensor of shape [n, v, seq_len (32*32), seq_len (32*32)]
            - plucker_coords: tensor of shape [n, v, dim (6), h (32), w (32)]
        """

        n_objects, n_views, c, h, w = x.shape
        x_in = x

        # note: if no context is given, cross-attention defaults to self-attention
        context, plucker = context[:-1], context[-1]
        context = [context]

        x = rearrange(x, "n v c h w -> (n v) c h w")
        x = self.norm(x)
        x = rearrange(x, "(n v) c h w -> n v c h w", v=n_views)

        # run input projection
        if not self.use_linear:
            # interpolate plucker to match x
            plucker = rearrange(plucker, "n v c h w -> (n v) c h w")
            plucker_interpolated = F.interpolate(
                plucker, size=x.shape[-2:], align_corners=False, mode="bilinear"
            )
            plucker_interpolated = rearrange(
                plucker_interpolated, "(n v) c h w -> n v c h w", v=n_views
            )

            # concat plucker to x
            x = torch.cat([x, plucker_interpolated], dim=2)
            x = rearrange(x, "n v c h w -> (n v) c h w")
            x = self.proj_in(x)
            x = rearrange(x, "(n v) c h w -> n v c h w", v=n_views)

        x = rearrange(x, "n v c h w -> n v (h w) c").contiguous()

        if self.use_linear:
            x = rearrange(x, "n v x c -> (n v) x c")
            x = self.proj_in(x)
            x = rearrange(x, "(n v) x c -> n v x c", v=n_views)

        # run the transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            _context = context[i]
            mask = None
            if isinstance(_context, (list, tuple)):
                try:
                    _context, mask = _context
                except:
                    _context = _context[0]
            x = block(x, context=_context, mask=mask)

        if x.isnan().any():
            breakpoint()

        # run output projection
        if self.use_linear:
            x = rearrange(x, "n v x c -> (n v) x c")
            x = self.proj_out(x)
            x = rearrange(x, "(n v) x c -> n v x c", v=n_views)

        x = rearrange(x, "n v (h w) c -> n v c h w", h=h, w=w).contiguous()

        if not self.use_linear:
            x = rearrange(x, "n v c h w -> (n v) c h w")
            x = self.proj_out(x)
            x = rearrange(x, "(n v) c h w -> n v c h w", v=n_views)

        return x + x_in


if __name__ == "__main__":
    spt_post = SPADTransformer(320, 8, 40, depth=1, context_dim=768).cuda()

    n_objects, n_views = 2, 4
    x = torch.randn(2, 4, 320, 32, 32).cuda()
    context = [
        torch.randn(n_objects, n_views, 77, 768).cuda(),
        torch.ones(
            n_objects, n_views * 32 * 32, n_views * 32 * 32, dtype=torch.bool
        ).cuda(),
        torch.randn(n_objects, n_views, 6, 32, 32).cuda(),
    ]
    x_post = spt_post(x, context=context)
