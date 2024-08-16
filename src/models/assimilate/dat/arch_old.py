#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:38:05 2020

@author: rfablet
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial, lru_cache
from timm.models.layers import DropPath
from timm.models.vision_transformer import trunc_normal_
import collections.abc
from einops import repeat, rearrange

class PatchEmbed(nn.Module):
    def __init__(self, img_size=None, patch_size=8, in_chans=13, embed_dim=768):
        super().__init__()

        if img_size is None:
            raise KeyError('img is None')

        patch_size = to_2tuple(patch_size)

        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

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

class SwinTransformer(nn.Module):
    """Implements the FuXi model as described in the paper,
    https://arxiv.org/abs/2301.10343

    Args:
        default_vars (list): list of default variables to be used for training
        img_size (list): image size of the input data
        patch_size (int): patch size of the input data
        embed_dim (int): embedding dimension
        depth (int): number of transformer layers
        num_blocks (int): number of fno blocks
        mlp_ratio (float): ratio of mlp hidden dimension to embedding dimension
        drop_path (float): stochastic depth rate
        drop_rate (float): dropout rate
        double_skip (bool): whether to use residual twice
    """

    def __init__(
        self,
        default_vars,
        img_size=[128, 256],
        window_size=8,
        patch_size=4,
        embed_dim=768,
        num_heads=16,
        depths=[2, 2, 2, 2],
        mlp_ratio=4,
        drop_path=0.2,
        drop_rate=0.2,
        attn_drop=0.,
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

        # variable tokenization: separate embedding layer for each input variable
        self.var_map = self.create_var_map()
        self.patch_embed = PatchEmbed(img_size, patch_size, len(default_vars), embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.in_norm = norm_layer(embed_dim, eps=1e-6)

        # positional embedding and lead time embedding
        self.lead_time_embed = nn.Sequential(
            nn.Linear(1, embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim, bias=True),
        )

        # --------------------------------------------------------------------------

        # MultiModal-Transformer backbone
        layers = []
        input_size = [sz // patch_size for sz in self.img_size]

        for i in range(self.num_layers):
            layer = SwinLayer(
                embed_dim,
                embed_dim,
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
            # self.add_module(f"norm{i}", norm_layer(6 * embed_dim, eps=1e-6))

        self.layers = nn.ModuleList(layers)

        # self.fpn = nn.Sequential(
        #     nn.Linear(6 * embed_dim * self.num_layers, 6 * embed_dim),
        #     nn.GELU(),
        # )
        # self.out_norm = nn.LayerNorm(embed_dim)
        # --------------------------------------------------------------------------

        # prediction head
        self.head = nn.Linear(embed_dim, len(self.default_vars) * patch_size ** 2)

        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # token embedding layer
        w = self.patch_embed.proj.weight.data
        trunc_normal_(w.view([w.shape[0], -1]), std=0.02)

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

    def forward_encoder(self, x: torch.Tensor, iter, variables, noise=None):
        # x: `[B, V, H, W]` shape.
        # tokenize each variable separately
        # (B, 8, H, W)
        h = self.patch_embed(x)

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
        lead_time_emb = self.lead_time_embed(iter)  # B, D

        # attention (swin or vit)
        # outs = []
        for i, blk in enumerate(self.layers):
            h = blk(h, lead_time_emb)
            # out = getattr(self, f"norm{i}")(h)
            # outs.append(out)
        # h = self.fpn(torch.cat(outs, dim=-1))

        # h = self.out_norm(h)

        return h

    def forward(self, grad, iter, variables, out_variables, noise=None):
        """Forward pass through the model.

        Args:
            x: `[B, Vi, H, W]` shape. Input weather/climate variables
            y: `[B, Vo, H, W]` shape. Target weather/climate variables
            lead_times: `[B]` shape. Forecasting lead times of each element of the batch.

        Returns:
            loss (list): Different metrics.
            preds (torch.Tensor): `[B, Vo, H, W]` shape. Predicted weather/climate variables.
        """
        iter_tensor = torch.from_numpy(iter * np.ones((1, 1))).to(grad.device, dtype=torch.float32) / 20
        out_transformers = self.forward_encoder(grad, iter_tensor, variables, noise)  # B, L, D

        preds = self.head(out_transformers)  # B, L, V*p*p

        preds = self.unpatchify(preds)
        out_var_ids = self.get_var_ids(tuple(out_variables), preds.device)
        preds = preds[:, out_var_ids]

        return preds

def compute_WeightedLoss(x2,w):
    #  fix normalizing factor ( Sum w = 1 != w~ bool index)
    if len(list(w.size()))>0:
        x2_msk = (x2 * w[None, :, None, None])[:, w>0, ...]
    else:
        x2_msk = x2[:, w==1, ...]
    x2_num = ~x2_msk.isnan() & ~x2_msk.isinf()
    if x2_num.sum() == 0:
        return torch.scalar_tensor(0., device=x2_num.device)
    # loss2 = x2_msk[x2_num].sum()
    loss2 = F.mse_loss(x2_msk[x2_num], torch.zeros_like(x2_msk[x2_num]))
    return loss2

def compute_spatio_temp_weighted_loss(x2, w):
    x2_w = (x2 * w[None, ...])
    non_zeros = (torch.ones_like(x2) * w[None, ...]) == 0.
    x2_num = ~x2_w.isnan() & ~x2_w.isinf() & ~non_zeros
    if x2_num.sum() == 0:
        return torch.scalar_tensor(0., device=x2_num.device)
    loss = F.mse_loss(x2_w[x2_num], torch.zeros_like(x2_w[x2_num]))
    return loss

# Modules for the definition of the norms for
# the observation and prior model
class Model_WeightedL2Norm(torch.nn.Module):
    def __init__(self):
        super(Model_WeightedL2Norm, self).__init__()

    def forward(self,x,w,eps=0.):
        loss_ = torch.nansum( x**2 , dim = -1)
        loss_ = torch.nansum( loss_ , dim = -1)
        if len(x.shape) == 5:
            loss_ = torch.nansum(loss_, dim=1)
        loss_ = torch.nansum( loss_ , dim = 0)
        loss_ = torch.nansum( loss_ * w )
        loss_ = loss_ / (torch.sum(~torch.isnan(x)) / w.shape[0] )

        return loss_

class Model_WeightedL1Norm(torch.nn.Module):
    def __init__(self):
        super(Model_WeightedL1Norm, self).__init__()

    def forward(self,x,w,eps):

        loss_ = torch.nansum( torch.sqrt( eps**2 + x**2 ) , dim = -1)
        loss_ = torch.nansum( loss_ , dim = -1)
        if len(x.shape) == 5:
            loss_ = torch.nansum(loss_, dim=1)
        loss_ = torch.nansum( loss_ , dim = 0)
        loss_ = torch.nansum( loss_ * w )
        loss_ = loss_ / (torch.sum(~torch.isnan(x)) / w.shape[0] )

        return loss_


# New module for the definition/computation of the variational cost
class Model_Var_Cost(nn.Module):
    def __init__(self ,m_NormObs, m_NormPhi, ShapeData, dim_obs=1,dim_obs_channel=0,dim_state=0):
        super(Model_Var_Cost, self).__init__()
        self.dim_obs_channel = dim_obs_channel
        self.dim_obs        = dim_obs
        if dim_state > 0 :
            self.dim_state      = dim_state
        else:
            self.dim_state      = ShapeData[0]

        # parameters for variational cost
        self.alphaObs    = torch.nn.Parameter(torch.Tensor(1. * np.ones((self.dim_obs,1))))
        self.alphaReg    = torch.nn.Parameter(torch.Tensor([1.]))
        if self.dim_obs_channel[0] == 0 :
            self.WObs           = torch.nn.Parameter(torch.Tensor(np.ones((self.dim_obs,ShapeData[0]))))
            self.dim_obs_channel  = ShapeData[0] * np.ones((self.dim_obs,))
        else:
            self.WObs            = torch.nn.Parameter(torch.Tensor(np.ones((self.dim_obs,np.max(self.dim_obs_channel)))))
        self.WReg    = torch.nn.Parameter(torch.Tensor(np.ones(self.dim_state,)))
        self.epsObs = torch.nn.Parameter(0.1 * torch.Tensor(np.ones((self.dim_obs,))))
        self.epsReg = torch.nn.Parameter(torch.Tensor([0.1]))

        self.normObs   = m_NormObs
        self.normPrior = m_NormPhi

    def forward(self, dx, dy):

        loss = self.alphaReg**2 * self.normPrior(dx,self.WReg**2,self.epsReg)
        if self.dim_obs == 1 :
            loss +=  self.alphaObs[0]**2 * self.normObs(dy,self.WObs[0,:]**2,self.epsObs[0])
        else:
            for kk in range(0,self.dim_obs):
                loss +=  (
                    self.alphaObs[kk]**2
                    * self.normObs(
                        dy[kk],
                        self.WObs[kk,0:dy[kk].size(1)]**2,
                        self.epsObs[kk]
                    )
                )

        return loss

class Model_H(torch.nn.Module):
    def __init__(self, shape_data):
        super(Model_H, self).__init__()
        self.dim_obs = 1
        self.dim_obs_channel = np.array(shape_data)

    def forward(self, x, y, mask):
        dyout = (x - y) * mask

        return dyout

# 4DVarNN Solver class using automatic differentiation for the computation of gradient of the variational cost
# input modules: operator phi_r, gradient-based update model m_Grad
# modules for the definition of the norm of the observation and prior terms given as input parameters
# (default norm (None) refers to the L2 norm)
# updated inner modles to account for the variational model module
class Solver_Grad_4DVarNN(nn.Module):
    NORMS = {
            'l1': Model_WeightedL1Norm,
            'l2': Model_WeightedL2Norm,
    }
    def __init__(self ,phi_r,mod_H, m_Grad, m_NormObs, m_NormPhi, shape_data, n_iter_grad, lead_time, dt):
        super(Solver_Grad_4DVarNN, self).__init__()
        self.phi_r         = phi_r

        if m_NormObs == None:
            m_NormObs =  Model_WeightedL2Norm()
        else:
            m_NormObs = self.NORMS[m_NormObs]()
        if m_NormPhi == None:
            m_NormPhi = Model_WeightedL2Norm()
        else:
            m_NormPhi = self.NORMS[m_NormPhi]()
        self.shape_data = shape_data
        self.model_H = mod_H
        self.model_Grad = m_Grad
        self.model_VarCost = Model_Var_Cost(m_NormObs, m_NormPhi, shape_data, mod_H.dim_obs, mod_H.dim_obs_channel)
        self.lead_time = lead_time
        self.dt = dt
        self.preds = []
        with torch.no_grad():
            self.n_grad = int(n_iter_grad)

    def forward(self, x, yobs, mask, vars, out_vars):
        return self.solve(x, yobs, mask, vars, out_vars)

    def solve(self, x_0, obs, mask, vars, out_vars, normgrad=None):
        x_k = torch.mul(x_0,1.)
        x_k_plus_1 = None
        for iter in range(self.n_grad):
            self.preds = []
            x_k_plus_1, normgrad = self.solver_step(x_0, x_k, obs, mask, iter, vars, out_vars, normgrad)

            x_k = torch.mul(x_k_plus_1,1.)

        return x_k_plus_1

    def solver_step(self, x_0, x_k, obs, mask, iter, vars, out_vars, normgrad=None):
        _, var_cost_grad = self.var_cost(x_0, x_k, obs, mask, vars, out_vars)
        if normgrad is None:
            normgrad_ = torch.sqrt(torch.mean(var_cost_grad**2, dim=(1,2,3), keepdim=True))
        else:
            normgrad_ = normgrad
        grad = self.model_Grad(var_cost_grad / normgrad_, iter, vars, out_vars)
        grad *= 1./ self.n_grad
        x_k_plus_1 = x_k - grad
        return x_k_plus_1, normgrad_

    def var_cost(self, x0, xk, yobs, mask, vars, out_vars):
        preds = self.forecast(xk, vars, out_vars)
        preds = torch.stack(self.preds, dim=1)
        dy = self.model_H(preds, yobs, mask)
        dx = xk - x0

        loss = self.model_VarCost(dx, dy)

        var_cost_grad = torch.autograd.grad(loss, xk, grad_outputs=torch.ones_like(loss), retain_graph=True)[0]
        return loss, var_cost_grad
    
    def forecast(self, x0, vars, out_vars):
        self.preds = []
        self.preds.append(x0)
        for i in range(1, self.lead_time // self.dt):
            if ((24 // self.dt) > 0) and (i % (24 // self.dt)) == 0:
                # Call the model for 24h forecast
                self.preds.append(self.phi_r(self.preds[i - 24 // self.dt],
                                             torch.from_numpy(24 * np.ones((1,1))).to(x0.device, dtype=torch.float32) / 100,
                                             vars,
                                             out_vars))
            elif ((12 // self.dt) > 0) and (i % (12 // self.dt)) == 0:
                # Call the model for 24h forecast
                self.preds.append(self.phi_r(self.preds[i - 12 // self.dt],
                                             torch.from_numpy(12 * np.ones((1,1))).to(x0.device, dtype=torch.float32) / 100,
                                             vars,
                                             out_vars))
            elif ((6 // self.dt) > 0) and (i % (6 // self.dt)) == 0:
                # Call the model for 24h forecast
                self.preds.append(self.phi_r(self.preds[i - 6 // self.dt],
                                             torch.from_numpy(6 * np.ones((1,1))).to(x0.device, dtype=torch.float32) / 100,
                                             vars,
                                             out_vars))
            elif ((3 // self.dt) > 0) and (i % (3 // self.dt)) == 0:
                # Call the model for 24h forecast
                self.preds.append(self.phi_r(self.preds[i - 3 // self.dt],
                                             torch.from_numpy(3 * np.ones((1,1))).to(x0.device, dtype=torch.float32) / 100,
                                             vars,
                                             out_vars))
        preds = torch.stack(self.preds, dim=1)
        return preds