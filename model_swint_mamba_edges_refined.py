import itertools
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from monai.networks.blocks import UnetrBasicBlock
from mamba_ssm import Mamba
from timm.models.layers import DropPath, trunc_normal_

from layers import fusion_prenorm, general_conv3d_prenorm


basic_dims = 8


def _ensure_3tuple(value):
    if isinstance(value, int):
        return (value, value, value)
    if len(value) != 3:
        raise ValueError(f"Expected a 3-tuple, got {value}.")
    return tuple(value)


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size=2,
        in_chans=1,
        embed_dim=48,
        norm_layer=nn.LayerNorm,
        spatial_dims=3,
    ) -> None:
        super().__init__()
        if spatial_dims != 3:
            raise ValueError("This refined implementation only supports 3D inputs.")

        self.patch_size = _ensure_3tuple(patch_size)
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError(f"PatchEmbed expects 5D input [B,C,D,H,W], got {tuple(x.shape)}.")

        _, _, d, h, w = x.shape
        pad_d = (self.patch_size[0] - d % self.patch_size[0]) % self.patch_size[0]
        pad_h = (self.patch_size[1] - h % self.patch_size[1]) % self.patch_size[1]
        pad_w = (self.patch_size[2] - w % self.patch_size[2]) % self.patch_size[2]
        if pad_d or pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))

        x = self.proj(x)
        if self.norm is not None:
            b, c, d, h, w = x.shape
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).reshape(b, c, d, h, w)
        return x


class PatchMergingV2(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm, spatial_dims=3):
        super().__init__()
        if spatial_dims != 3:
            raise ValueError("This refined implementation only supports 3D inputs.")
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError(f"PatchMergingV2 expects [B,D,H,W,C], got {tuple(x.shape)}.")

        b, d, h, w, c = x.shape
        if c != self.dim:
            raise ValueError(f"Channel mismatch: expected {self.dim}, got {c}.")

        if (d % 2) or (h % 2) or (w % 2):
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))

        x = torch.cat(
            [x[:, i::2, j::2, k::2, :] for i, j, k in itertools.product(range(2), range(2), range(2))],
            dim=-1,
        )
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchMerging(PatchMergingV2):
    def forward(self, x):
        if x.dim() != 5:
            raise ValueError(f"PatchMerging expects [B,D,H,W,C], got {tuple(x.shape)}.")

        b, d, h, w, c = x.shape
        if c != self.dim:
            raise ValueError(f"Channel mismatch: expected {self.dim}, got {c}.")

        if (d % 2) or (h % 2) or (w % 2):
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))

        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 0::2, 1::2, :]
        x5 = x[:, 1::2, 1::2, 0::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], dim=-1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}


def window_partition(x, window_size):
    b, d, h, w, c = x.shape
    x = x.view(
        b,
        d // window_size[0],
        window_size[0],
        h // window_size[1],
        window_size[1],
        w // window_size[2],
        window_size[2],
        c,
    )
    windows = (
        x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(
            -1, window_size[0] * window_size[1] * window_size[2], c
        )
    )
    return windows


def window_reverse(windows, window_size, dims):
    b, d, h, w = dims
    x = windows.view(
        b,
        d // window_size[0],
        h // window_size[1],
        w // window_size[2],
        window_size[0],
        window_size[1],
        window_size[2],
        -1,
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    use_shift_size = list(shift_size) if shift_size is not None else None
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if use_shift_size is not None:
                use_shift_size[i] = 0
    if use_shift_size is None:
        return tuple(use_window_size)
    return tuple(use_window_size), tuple(use_shift_size)


def compute_mask(dims, window_size, shift_size, device):
    d, h, w = dims
    cnt = 0
    img_mask = torch.zeros((1, d, h, w, 1), device=device)
    for d_slice in (
        slice(-window_size[0]),
        slice(-window_size[0], -shift_size[0]),
        slice(-shift_size[0], None),
    ):
        for h_slice in (
            slice(-window_size[1]),
            slice(-window_size[1], -shift_size[1]),
            slice(-shift_size[1], None),
        ):
            for w_slice in (
                slice(-window_size[2]),
                slice(-window_size[2], -shift_size[2]),
                slice(-shift_size[2], None),
            ):
                img_mask[:, d_slice, h_slice, w_slice, :] = cnt
                cnt += 1

    mask_windows = window_partition(img_mask, window_size).squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
    return attn_mask


class Mlp(nn.Module):
    def __init__(self, hidden_size, mlp_dim, dropout_rate=0.0):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, mlp_dim or hidden_size)
        self.linear2 = nn.Linear(mlp_dim or hidden_size, hidden_size)
        self.fn = nn.GELU()
        self.drop1 = nn.Dropout(dropout_rate)
        self.drop2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fn(self.linear1(x))
        x = self.drop1(x)
        x = self.linear2(x)
        x = self.drop2(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                num_heads,
            )
        )
        coords_d = torch.arange(window_size[0])
        coords_h = torch.arange(window_size[1])
        coords_w = torch.arange(window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * window_size[2] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:n, :n].reshape(-1)
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn).to(v.dtype)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size,
        shift_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = norm_layer(dim)
        self.normtoken = norm_layer(dim)
        self.attn = WindowAttention(dim, num_heads, window_size, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop)

    def forward_part1(self, x, mask_matrix, m_token=None):
        x = self.norm1(x) if m_token is None else self.norm1(x) + self.normtoken(m_token)
        b, d, h, w, c = x.shape
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        pad_d = (window_size[0] - d % window_size[0]) % window_size[0]
        pad_h = (window_size[1] - h % window_size[1]) % window_size[1]
        pad_w = (window_size[2] - w % window_size[2]) % window_size[2]
        if pad_d or pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))

        _, dp, hp, wp, _ = x.shape
        dims = [b, dp, hp, wp]
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)

        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=shift_size, dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d or pad_h or pad_w:
            x = x[:, :d, :h, :w, :].contiguous()
        return x

    def forward(self, x, mask_matrix, m_token=None):
        shortcut = x
        x = self.forward_part1(x, mask_matrix, m_token)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size,
        drop_path,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for _ in window_size)
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=self.window_size,
                    shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.downsample = None
        if callable(downsample):
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, spatial_dims=3)

    def forward(self, x, m_token=None):
        if x.dim() != 5:
            raise ValueError(f"BasicLayer expects [B,C,D,H,W], got {tuple(x.shape)}.")

        b, c, d, h, w = x.shape
        x = rearrange(x, "b c d h w -> b d h w c")
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        dp = int(np.ceil(d / window_size[0])) * window_size[0]
        hp = int(np.ceil(h / window_size[1])) * window_size[1]
        wp = int(np.ceil(w / window_size[2])) * window_size[2]
        attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x.device)

        if m_token is not None:
            if m_token.shape[-1] != c:
                raise ValueError(f"m_token channel mismatch: expected {c}, got {m_token.shape[-1]}.")
            m_token = m_token.expand(b, -1, -1, -1, c)

        for blk in self.blocks:
            x = blk(x, attn_mask, m_token)

        x = x.view(b, d, h, w, -1)
        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, "b d h w c -> b c d h w")
        return x


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand, bimamba_type="v2")

    def forward(self, x):
        b, c = x.shape[:2]
        if c != self.dim:
            raise ValueError(f"MambaLayer expects channel={self.dim}, got {c}.")
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(b, c, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(b, c, *img_dims)
        return out


class MlpChannel(nn.Module):
    def __init__(self, hidden_size, mlp_dim):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        if self.data_format != "channels_first":
            raise NotImplementedError
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None, None] * x + self.bias[:, None, None, None]


class MambaBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depth=2, downsample=True):
        super().__init__()
        self.downsample = downsample
        self.out_channels = out_channels
        if self.downsample:
            self.downsample_layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2),
                LayerNorm(out_channels, eps=1e-6, data_format="channels_first"),
            )
        elif in_channels != out_channels:
            raise ValueError("When downsample=False, in_channels must equal out_channels.")

        self.mamba_layers = nn.Sequential(*[MambaLayer(dim=out_channels) for _ in range(depth)])
        self.norm = LayerNorm(out_channels, eps=1e-6, data_format="channels_first")
        self.mlp = MlpChannel(out_channels, out_channels * 4)

    def forward(self, x):
        if self.downsample:
            x = self.downsample_layer(x)
        if x.shape[1] != self.out_channels:
            raise ValueError(f"MambaBlock expects {self.out_channels} channels, got {x.shape[1]}.")
        x = self.mamba_layers(x)
        x = self.norm(x)
        x = self.mlp(x)
        return x


class SwinTransformer(nn.Module):
    def __init__(
        self,
        in_chans,
        embed_dim,
        window_size,
        patch_size,
        depths,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        patch_norm=False,
        feat_size=(8, 24, 48, 96, 192),
        norm_name="instance",
        spatial_dims=3,
        downsample="merging",
    ):
        super().__init__()
        if len(depths) != 3 or len(num_heads) != 3:
            raise ValueError(
                f"This refined SwinTransformer keeps 3 stages. Expected len(depths)==len(num_heads)==3, got {len(depths)} and {len(num_heads)}."
            )

        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = _ensure_3tuple(window_size)
        self.patch_size = _ensure_3tuple(patch_size)
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
            spatial_dims=spatial_dims,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        down_sample_mod = MERGING_MODE[downsample] if isinstance(downsample, str) else downsample

        self.layers = nn.ModuleList()
        for i_layer in range(3):
            self.layers.append(
                BasicLayer(
                    dim=int(embed_dim * 2 ** i_layer),
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=self.window_size,
                    drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                    downsample=down_sample_mod,
                )
            )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_chans,
            out_channels=feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.enc_mamba = nn.Sequential(
            MambaBlock(in_channels=feat_size[4], out_channels=feat_size[4], depth=2, downsample=False),
            MambaBlock(in_channels=feat_size[4], out_channels=feat_size[4], depth=2, downsample=False),
            MambaBlock(in_channels=feat_size[4], out_channels=feat_size[4], depth=2, downsample=False),
            MambaBlock(in_channels=feat_size[4], out_channels=feat_size[4], depth=2, downsample=False),
        )
        self.dec_mamba = nn.Sequential(
            MambaBlock(in_channels=feat_size[4], out_channels=feat_size[4], depth=2, downsample=False),
            MambaBlock(in_channels=feat_size[4], out_channels=feat_size[4], depth=2, downsample=False),
            MambaBlock(in_channels=feat_size[4], out_channels=feat_size[4], depth=2, downsample=False),
            MambaBlock(in_channels=feat_size[4], out_channels=feat_size[4], depth=2, downsample=False),
        )

    def proj_out(self, x, normalize=False):
        if not normalize:
            return x
        if x.dim() != 5:
            raise ValueError(f"proj_out expects [B,C,D,H,W], got {tuple(x.shape)}.")
        n, ch, d, h, w = x.shape
        x = rearrange(x, "n c d h w -> n d h w c")
        x = F.layer_norm(x, [ch])
        x = rearrange(x, "n d h w c -> n c d h w")
        return x

    def forward(self, x, normalize=True):
        if x.dim() != 5 or x.shape[1] != 1:
            raise ValueError(f"SwinTransformer expects [B,1,D,H,W], got {tuple(x.shape)}.")

        enc0 = self.encoder1(x)
        x0 = self.pos_drop(self.patch_embed(x))
        x0_out = self.proj_out(x0, normalize)

        x1 = self.layers[0](x0.contiguous())
        x1_out = self.proj_out(x1, normalize)

        x2 = self.layers[1](x1.contiguous())
        x2_out = self.proj_out(x2, normalize)

        x3 = self.layers[2](x2.contiguous())
        x3_out = self.dec_mamba(self.enc_mamba(x3))
        return [enc0, x0_out, x1_out, x2_out, x3_out]


class MultiClassBEM3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.modality_fusion = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(in_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.edge_head = nn.Sequential(
            general_conv3d_prenorm(in_channels, in_channels // 2, k_size=3, padding=1),
            nn.Conv3d(in_channels // 2, 3, kernel_size=1),
        )
        self.gate_conv = nn.Sequential(nn.Conv3d(3, in_channels, kernel_size=1), nn.Sigmoid())
        self.refine_conv = general_conv3d_prenorm(in_channels * 2, in_channels, k_size=3, padding=1)

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError(f"MultiClassBEM3D expects [B,C,D,H,W], got {tuple(x.shape)}.")
        feat_fused = self.modality_fusion(x)
        edge_maps = self.edge_head(feat_fused)
        gate = self.gate_conv(edge_maps)
        edge_enhanced = feat_fused * gate
        enhanced_feat = self.refine_conv(torch.cat([feat_fused, edge_enhanced], dim=1))
        return enhanced_feat, edge_maps


class Decoder_BEM_Fuse(nn.Module):
    def __init__(self, num_cls=4, filters=(32, 96, 192, 384, 768)):
        super().__init__()
        c1, c2, c3, c4, c5 = filters
        self.num_cls = num_cls

        self.bem5 = MultiClassBEM3D(c5)
        self.bem4 = MultiClassBEM3D(c4)
        self.bem3 = MultiClassBEM3D(c3)
        self.bem2 = MultiClassBEM3D(c2)
        self.bem1 = MultiClassBEM3D(c1)

        self.RFM5 = fusion_prenorm(in_channel=c5, num_cls=num_cls)
        self.RFM4 = fusion_prenorm(in_channel=c4, num_cls=num_cls)
        self.RFM3 = fusion_prenorm(in_channel=c3, num_cls=num_cls)
        self.RFM2 = fusion_prenorm(in_channel=c2, num_cls=num_cls)
        self.RFM1 = fusion_prenorm(in_channel=c1, num_cls=num_cls)

        self.d4_c1 = general_conv3d_prenorm(c5, c4, pad_type="reflect")
        self.d4_c2 = general_conv3d_prenorm(c4 + c4, c4, pad_type="reflect")
        self.d4_out = general_conv3d_prenorm(c4, c4, k_size=1, padding=0, pad_type="reflect")

        self.d3_c1 = general_conv3d_prenorm(c4, c3, pad_type="reflect")
        self.d3_c2 = general_conv3d_prenorm(c3 + c3, c3, pad_type="reflect")
        self.d3_out = general_conv3d_prenorm(c3, c3, k_size=1, padding=0, pad_type="reflect")

        self.d2_c1 = general_conv3d_prenorm(c3, c2, pad_type="reflect")
        self.d2_c2 = general_conv3d_prenorm(c2 + c2, c2, pad_type="reflect")
        self.d2_out = general_conv3d_prenorm(c2, c2, k_size=1, padding=0, pad_type="reflect")

        self.d1_c1 = general_conv3d_prenorm(c2, c1, pad_type="reflect")
        self.d1_c2 = general_conv3d_prenorm(c1 + c1, c1, pad_type="reflect")
        self.d1_out = general_conv3d_prenorm(c1, c1, k_size=1, padding=0, pad_type="reflect")

        self.seg_d4 = nn.Conv3d(c5, num_cls, kernel_size=1, bias=True)
        self.seg_d3 = nn.Conv3d(c4, num_cls, kernel_size=1, bias=True)
        self.seg_d2 = nn.Conv3d(c3, num_cls, kernel_size=1, bias=True)
        self.seg_d1 = nn.Conv3d(c2, num_cls, kernel_size=1, bias=True)
        self.seg_layer = nn.Conv3d(c1, num_cls, kernel_size=1, bias=True)

        self.up2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode="trilinear", align_corners=True)

    @staticmethod
    def _assert_same_spatial(a, b, name_a, name_b):
        if a.shape[2:] != b.shape[2:]:
            raise ValueError(f"Spatial mismatch between {name_a} {tuple(a.shape)} and {name_b} {tuple(b.shape)}.")

    def forward(self, x1, x2, x3, x4, x5, return_edges=False, apply_softmax=False):
        if not all(t.dim() == 5 for t in (x1, x2, x3, x4, x5)):
            raise ValueError("Decoder_BEM_Fuse expects five 5D tensors [B,C,D,H,W].")

        edge_maps_list = []

        de_x5, edge5 = self.bem5(x5)
        de_x5 = self.RFM5(de_x5)
        logit_d4 = self.seg_d4(de_x5)
        edge_maps_list.append(edge5)

        de_x5_up = self.d4_c1(self.up2(de_x5))
        de_x4, edge4 = self.bem4(x4)
        de_x4 = self.RFM4(de_x4)
        self._assert_same_spatial(de_x4, de_x5_up, "de_x4", "de_x5_up")
        de_x4 = self.d4_out(self.d4_c2(torch.cat((de_x4, de_x5_up), dim=1)))
        logit_d3 = self.seg_d3(de_x4)
        edge_maps_list.append(edge4)

        de_x4_up = self.d3_c1(self.up2(de_x4))
        de_x3, edge3 = self.bem3(x3)
        de_x3 = self.RFM3(de_x3)
        self._assert_same_spatial(de_x3, de_x4_up, "de_x3", "de_x4_up")
        de_x3 = self.d3_out(self.d3_c2(torch.cat((de_x3, de_x4_up), dim=1)))
        logit_d2 = self.seg_d2(de_x3)
        edge_maps_list.append(edge3)

        de_x3_up = self.d2_c1(self.up2(de_x3))
        de_x2, edge2 = self.bem2(x2)
        de_x2 = self.RFM2(de_x2)
        self._assert_same_spatial(de_x2, de_x3_up, "de_x2", "de_x3_up")
        de_x2 = self.d2_out(self.d2_c2(torch.cat((de_x2, de_x3_up), dim=1)))
        logit_d1 = self.seg_d1(de_x2)
        edge_maps_list.append(edge2)

        de_x2_up = self.d1_c1(self.up2(de_x2))
        de_x1, edge1 = self.bem1(x1)
        de_x1 = self.RFM1(de_x1)
        self._assert_same_spatial(de_x1, de_x2_up, "de_x1", "de_x2_up")
        de_x1 = self.d1_out(self.d1_c2(torch.cat((de_x1, de_x2_up), dim=1)))
        edge_maps_list.append(edge1)

        logits = self.seg_layer(de_x1)
        deep_logits = (
            self.up2(logit_d1),
            self.up4(logit_d2),
            self.up8(logit_d3),
            self.up16(logit_d4),
        )

        if apply_softmax:
            pred = torch.softmax(logits, dim=1)
            deep_preds = tuple(torch.softmax(t, dim=1) for t in deep_logits)
            if return_edges:
                return pred, deep_preds, list(reversed(edge_maps_list))
            return pred, deep_preds

        if return_edges:
            return logits, deep_logits, list(reversed(edge_maps_list))
        return logits, deep_logits


class MaskModal(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask):
        if x.dim() != 6:
            raise ValueError(f"MaskModal expects [B,K,C,D,H,W], got {tuple(x.shape)}.")
        if mask.dim() != 2:
            raise ValueError(f"MaskModal mask expects [B,K], got {tuple(mask.shape)}.")

        b, k, c, d, h, w = x.shape
        if mask.shape != (b, k):
            raise ValueError(f"Mask shape {tuple(mask.shape)} must match modal dims ({b}, {k}).")

        mask = mask.to(device=x.device, dtype=x.dtype)
        mask = mask.view(b, k, 1, 1, 1, 1)
        x = x * mask
        fused = x.reshape(b, k * c, d, h, w)
        return fused


class Model(nn.Module):
    def __init__(
        self,
        num_cls=4,
        img_size=128,
        in_channels=1,
        normalize=True,
        depths=(2, 2, 2),
        hidden_size=384,
        model_num_heads=(3, 6, 12),
        norm_name="instance",
        feature_size=24,
        spatial_dims=3,
        use_checkpoint=False,
    ):
        super().__init__()
        img_size = (img_size,) * spatial_dims
        patch_size = (2,) * spatial_dims
        window_size = (7,) * spatial_dims
        for m, p in zip(img_size, patch_size):
            for i in range(5):
                if m % np.power(p, i + 1) != 0:
                    raise ValueError("img_size should be divisible by stage-wise image resolution.")

        self.normalize = normalize
        self.hidden_size = hidden_size
        self.num_modalities = 4

        self.swinEncoder = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=model_num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=nn.LayerNorm,
            spatial_dims=spatial_dims,
            norm_name=norm_name,
        )
        self.masker = MaskModal()
        self.decoder_fuse = Decoder_BEM_Fuse(num_cls=num_cls, filters=(32, 96, 192, 384, 768))

    def _encode_modalities_parallel(self, x):
        if x.dim() != 5 or x.shape[1] != self.num_modalities:
            raise ValueError(f"Model expects [B,{self.num_modalities},D,H,W], got {tuple(x.shape)}.")

        b, m, d, h, w = x.shape
        x_parallel = x.reshape(b * m, 1, d, h, w)
        encoded = self.swinEncoder(x_parallel, self.normalize)

        modal_features: List[torch.Tensor] = []
        for feat in encoded:
            if feat.dim() != 5:
                raise ValueError(f"Encoded feature must be 5D, got {tuple(feat.shape)}.")
            _, c, fd, fh, fw = feat.shape
            modal_features.append(feat.reshape(b, m, c, fd, fh, fw))
        return modal_features

    def forward(self, x, mask, return_edges=False, apply_softmax=None):
        if apply_softmax is None:
            apply_softmax = not self.training
        if mask.dtype != torch.bool:
            mask = mask.bool()

        features = self._encode_modalities_parallel(x)
        if len(features) != 5:
            raise RuntimeError(f"Expected 5 encoder features, got {len(features)}.")

        x1, x2, x3, x4, x5 = [self.masker(feat, mask) for feat in features]

        return self.decoder_fuse(
            x1,
            x2,
            x3,
            x4,
            x5,
            return_edges=return_edges,
            apply_softmax=apply_softmax,
        )
