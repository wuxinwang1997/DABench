# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import partial, lru_cache
import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from timm.models.vision_transformer import trunc_normal_
import collections.abc
from einops import repeat, rearrange
import torch.nn.functional as F
from src.utils.model_utils import load_constant

def exists(val):
    return val is not None

def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)

class PeriodicConv2d(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert max(self.padding) > 0

    def forward(self, x):
        x = F.pad(x, (self.padding[1], self.padding[1], 0, 0), mode="circular")
        x = F.pad(x, (0, 0, self.padding[0], self.padding[0]), mode="constant", value=0)
        x = F.conv2d(x, self.weight, self.bias, self.stride, 0, self.dilation, self.groups)
        return x

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x

class GeGLUFFN(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        inner_dim = int(hidden_features * (2 / 3))
        self.fc1 = nn.Linear(in_features, inner_dim * 2, bias=False)
        self.act = GEGLU()
        self.fc2 = nn.Linear(inner_dim, out_features, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0],W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttentionV2(nn.Module):
    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        attn_drop=0.,
        proj_drop=0.,
        pretrained_window_size=[0, 0]
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)

        relative_coords_table = torch.stack(
            torch.meshgrid(
                [relative_coords_h, relative_coords_w], indexing='ij')
        ).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2

        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :,
                                  0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :,
                                  1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(
            [coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - \
            coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - \
            1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index",
                             relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias,
                                  torch.zeros_like(self.v_bias, requires_grad=False),
                                  self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # cosine attention
        attn = (F.normalize(q, dim=-1) @
                F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(
            torch.tensor(1. / 0.01)).to(self.logit_scale.device)).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(
            self.relative_coords_table.to(x)).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = mask.to(x)
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.out_proj(x)
        x = self.proj_drop(x)
        return x

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class SwinBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        input_size,
        window_size=7,
        shift_size=0,
        mask_type='h',
        mlp_ratio=4.,
        qkv_bias=True,
        drop=0.,
        drop_path=0.,
        attn_drop=0.,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.dim = dim
        self.input_size = input_size
        self.num_heads = num_heads
        self.window_size = list(to_2tuple(window_size))
        self.shift_size = list(to_2tuple(shift_size))
        self.mlp_ratio = mlp_ratio

        if self.input_size[0] <= self.window_size[0]:
            self.shift_size[0] = 0
            self.window_size[0] = self.input_size[0]

        if self.input_size[1] <= self.window_size[1]:
            self.shift_size[1] = 0
            self.window_size[1] = self.input_size[1]

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)

        self.attn = WindowAttentionV2(dim,
                                      window_size=self.window_size,
                                      num_heads=num_heads,
                                      qkv_bias=qkv_bias,
                                      attn_drop=attn_drop,
                                      proj_drop=drop)

        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = GeGLUFFN(
            in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        if max(self.shift_size) > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_size
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size[0]),
                        slice(-self.window_size[0], -self.shift_size[0]),
                        slice(-self.shift_size[0], None))
            w_slices = (slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], -self.shift_size[1]),
                        slice(-self.shift_size[1], None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    if mask_type == 'h':
                        img_mask[:, h, :, :] = cnt
                    elif mask_type == 'w':
                        img_mask[:, :, w, :] = cnt
                    else:
                        img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )

        self.register_buffer("attn_mask", attn_mask)

    def swin_attn(self, x):
        H, W = self.input_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        # cyclic shift
        if max(self.shift_size) > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        # nW*B, window_size, window_size, C
        x_windows = window_partition(shifted_x, self.window_size)
        # nW*B, window_size*window_size, C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)

        # W-MSA/SW-MSA
        # nW*B, window_size*window_size, C
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if max(self.shift_size) > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        return x

    def forward(self, x, dt):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(dt).chunk(6, dim=-1)
        x = x + self.drop_path1(gate_msa.unsqueeze(1) * self.swin_attn(modulate(self.norm1(x), shift_msa, scale_msa)))
        x = x + self.drop_path2(gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)))
        return x


norm_layer = partial(nn.LayerNorm, eps=1e-6)


class SwinLayer(nn.Module):

    def __init__(
        self,
        in_chans,
        embed_dim,
        input_size,
        window_size,
        depth=4,
        num_heads=8,
        mlp_ratio=4.,
        drop=0.,
        drop_path=0.,
        attn_drop=0.,
    ):

        super().__init__()

        self.depth = depth
        self.input_size = input_size

        self.blocks = nn.ModuleList()

        for i in range(depth):
            blk = SwinBlock(
                dim=embed_dim,
                input_size=input_size,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path,
                attn_drop=attn_drop,
                norm_layer=norm_layer,
            )
            self.blocks.append(blk)

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, h, temb):
        for i, blk in enumerate(self.blocks):
            h = blk(h, temb)
        return h

class Sformer(nn.Module):
    def __init__(
        self,
        default_vars,
        img_size=[128, 256],
        window_size=8,
        patch_size=4,
        surface_vars=8,
        pressure_level=9,
        embed_dim=768,
        num_heads=16,
        depths=[4, 4, 4, 4],
        mlp_ratio=4,
        drop_path=0.2,
        drop_rate=0.2,
        attn_drop=0.,
        const_dir="../../data/train_pred",
    ):
        super().__init__()

        # TODO: remove time_history parameter
        self.img_size = img_size
        self.patch_size = patch_size
        self.default_vars = default_vars
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.c = len(self.default_vars)
        self.h = self.img_size[0] // patch_size
        self.w = self.img_size[1] // patch_size
        self.feat_size = [sz // patch_size for sz in img_size]
        self.embed_dim = embed_dim
        self.num_layers = len(depths)
        self.surface_vars = int(surface_vars)
        self.pressure_level = int(pressure_level)
        self.constant = torch.from_numpy(load_constant(const_dir))

        # variable tokenization: separate embedding layer for each input variable
        self.var_map = self.create_var_map()
        self.patch_embed_s = nn.Conv2d(in_channels=self.surface_vars,
                                     out_channels=embed_dim,
                                     kernel_size=patch_size,
                                     stride=patch_size)
        self.patch_embed_z = nn.Conv2d(in_channels=self.pressure_level,
                                     out_channels=embed_dim,
                                     kernel_size=patch_size,
                                     stride=patch_size)
        self.patch_embed_u = nn.Conv2d(in_channels=self.pressure_level,
                                     out_channels=embed_dim,
                                     kernel_size=patch_size,
                                     stride=patch_size)
        self.patch_embed_v = nn.Conv2d(in_channels=self.pressure_level,
                                     out_channels=embed_dim,
                                     kernel_size=patch_size,
                                     stride=patch_size)
        self.patch_embed_t = nn.Conv2d(in_channels=self.pressure_level,
                                     out_channels=embed_dim,
                                     kernel_size=patch_size,
                                     stride=patch_size)
        self.patch_embed_r = nn.Conv2d(in_channels=self.pressure_level,
                                     out_channels=embed_dim,
                                     kernel_size=patch_size,
                                     stride=patch_size)
        self.num_patches = self.h * self.w
        self.in_norm = norm_layer(6 * embed_dim, eps=1e-6)

        # positional embedding and lead time embedding
        self.lead_time_embed = nn.Sequential(
            nn.Linear(1, 6 * embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(6 * embed_dim, 6 * embed_dim, bias=True),
        )

        # --------------------------------------------------------------------------

        # MultiModal-Transformer backbone
        layers = []
        input_size = [sz // patch_size for sz in self.img_size]

        for i in range(self.num_layers):
            layer = SwinLayer(
                6 * embed_dim,
                6 * embed_dim,
                input_size,
                window_size,
                depth=depths[i],
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=drop_path,
                attn_drop=attn_drop,
            )
            layers.append(layer)
            self.add_module(f"norm{i}", norm_layer(6 * embed_dim, eps=1e-6))

        self.layers = nn.ModuleList(layers)

        self.fpn = nn.Sequential(
            nn.Linear(6 * embed_dim * self.num_layers, 6 * embed_dim),
            nn.GELU(),
        )
        self.out_norm = nn.LayerNorm(6 * embed_dim)
        # --------------------------------------------------------------------------

        # prediction head
        self.head = nn.Linear(6 * embed_dim, len(self.default_vars) * patch_size ** 2)

        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # token embedding layer
        ws = self.patch_embed_s.weight.data
        trunc_normal_(ws.view([ws.shape[0], -1]), std=0.02)
        wz = self.patch_embed_z.weight.data
        trunc_normal_(wz.view([wz.shape[0], -1]), std=0.02)
        wu = self.patch_embed_u.weight.data
        trunc_normal_(wu.view([wu.shape[0], -1]), std=0.02)
        wv = self.patch_embed_v.weight.data
        trunc_normal_(wv.view([wv.shape[0], -1]), std=0.02)
        wt = self.patch_embed_t.weight.data
        trunc_normal_(wt.view([wt.shape[0], -1]), std=0.02)
        wr = self.patch_embed_r.weight.data
        trunc_normal_(wr.view([wr.shape[0], -1]), std=0.02)
        
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def create_var_map(self):
        # TODO: create a mapping from var --> idx
        var_map = {}
        idx = 0
        for var in self.default_vars:
            var_map[var] = idx
            idx += 1
        return var_map

    @lru_cache(maxsize=None)
    def get_var_ids(self, vars, device):
        ids = np.array([self.var_map[var] for var in vars])
        return torch.from_numpy(ids).to(device)

    def unpatchify(self, x: torch.Tensor):
        """
        x: (B, L, V * patch_size**2)
        return imgs: (B, V, H, W)
        """
        x = x.reshape(shape=(x.shape[0], self.h, self.w, self.patch_size, self.patch_size, self.c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], self.c, self.h * self.patch_size, self.w * self.patch_size))
        return imgs

    def forward_encoder(self, x: torch.Tensor, lead_times, variables, noise=None):
        # x: `[B, V, H, W]` shape.
        # tokenize each variable separately
        # (B, 8, H, W)
        surface = x[:, :self.surface_vars]
        # (B, 13, H, W)
        z = x[:, self.surface_vars:int(self.surface_vars+self.pressure_level)]
        u = x[:, int(self.surface_vars+self.pressure_level):int(self.surface_vars+2*self.pressure_level)]
        v = x[:, int(self.surface_vars+2*self.pressure_level):int(self.surface_vars+3*self.pressure_level)]
        t = x[:, int(self.surface_vars+3*self.pressure_level):int(self.surface_vars+4*self.pressure_level)]
        q = x[:, int(self.surface_vars+4*self.pressure_level):int(self.surface_vars+5*self.pressure_level)]
        # (B, C, H/p, W/p)
        in_s = self.patch_embed_s(surface)
        in_z = self.patch_embed_z(z)
        in_u = self.patch_embed_u(u)
        in_v = self.patch_embed_v(v)
        in_t = self.patch_embed_t(t)
        in_q = self.patch_embed_r(q)
        # (B, 6C, H/p. W/p)
        h = torch.concat([in_s, in_z, in_u, in_v, in_t, in_q], dim=1)
        # (B, HW/p**2, 6C)
        h = h.flatten(2).transpose(1, 2)
        h = self.in_norm(h)

        if exists(noise):
            noise = F.interpolate(
                noise,
                size=self.feat_size,
                mode="bilinear",
                align_corners=False,
            )
            noise = rearrange(noise, 'n c h w -> n (h w) c')
            h = h + noise.to(h)

        # add lead time embedding
        lead_time_emb = self.lead_time_embed(lead_times)  # B, D

        # attention (swin or vit)
        outs = []
        for i, blk in enumerate(self.layers):
            h = blk(h, lead_time_emb)
            out = getattr(self, f"norm{i}")(h)
            outs.append(out)
        h = self.fpn(torch.cat(outs, dim=-1))

        h = self.out_norm(h)

        return h

    def forward(self, x, lead_times, variables, out_variables, noise=None):
        """Forward pass through the model.

        Args:
            x: `[B, Vi, H, W]` shape. Input weather/climate variables
            y: `[B, Vo, H, W]` shape. Target weather/climate variables
            lead_times: `[B]` shape. Forecasting lead times of each element of the batch.

        Returns:
            loss (list): Different metrics.
            preds (torch.Tensor): `[B, Vo, H, W]` shape. Predicted weather/climate variables.
        """
        constant = torch.repeat_interleave(self.constant, x.shape[0], dim=0).to(x.device, dtype=x.dtype)
        out_transformers = self.forward_encoder(torch.concat([constant, x], dim=1), lead_times, variables, noise)  # B, L, D

        preds = self.head(out_transformers)  # B, L, V*p*p

        preds = self.unpatchify(preds)
        out_var_ids = self.get_var_ids(tuple(out_variables), preds.device)
        preds = preds[:, out_var_ids] + x

        return preds
