import argparse
import ast
from typing import Optional, Union

import numpy as np
import timm
import torch
import torch.nn as nn
from earthnet_models_pytorch.model.layer_utils import inverse_permutation
from earthnet_models_pytorch.utils import str2bool
from torch.jit import Final
from torch.nn import functional as F


class Attention(nn.Module):
    # https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
    fast_attn: Final[bool]

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fast_attn = hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        )  # FIXME

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fast_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        drop=0.0,
        attn_drop=0.0,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


def get_sinusoid_encoding_table(positions, d_hid, T=1000):
    """Sinusoid position encoding table
    positions: int or list of integer, if int range(positions)"""

    if isinstance(positions, int):
        positions = list(range(positions))

    def cal_angle(position, hid_idx):
        return position / np.power(T, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in positions])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table)


class PVT_embed(nn.Module):

    def __init__(self, in_channels, out_channels, pretrained=True, frozen=False):
        super().__init__()

        self.pvt = timm.create_model(
            "pvt_v2_b0.in1k",
            pretrained=pretrained,
            features_only=True,
            in_chans=in_channels,
        )
        if frozen:
            timm.utils.freeze(self.pvt)
        self.pvt_project = nn.Conv2d(
            in_channels=512,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            bias=True,
        )

    def forward(self, x):

        B, T, C, H, W = x.shape

        x_feats = self.pvt(x.reshape(B * T, C, H, W))

        x_feats = [F.interpolate(feat, size=x_feats[0].shape[-2:]) for feat in x_feats]

        x = self.pvt_project(torch.cat(x_feats, dim=1))

        _, C, H, W = x.shape

        # Patchify

        x_patches = (
            x.reshape(B, T, C, H, W).permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)
        )

        return x_patches


class ContextFormer(nn.Module):

    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        if self.hparams.pvt:
            self.embed_images = PVT_embed(
                in_channels=self.hparams.n_image,
                out_channels=self.hparams.n_hidden,
                pretrained=True,
                frozen=self.hparams.pvt_frozen,
            )
        else:
            self.embed_images = Mlp(
                in_features=self.hparams.n_image
                * self.hparams.patch_size
                * self.hparams.patch_size,
                hidden_features=self.hparams.n_hidden,
                out_features=self.hparams.n_hidden,
            )

        self.embed_weather = Mlp(
            in_features=self.hparams.n_weather,
            hidden_features=self.hparams.n_hidden,
            out_features=self.hparams.n_hidden,
        )

        self.mask_token = nn.Parameter(torch.zeros(self.hparams.n_hidden))

        self.blocks = nn.ModuleList(
            [
                Block(
                    self.hparams.n_hidden,
                    self.hparams.n_heads,
                    self.hparams.mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(self.hparams.depth)
            ]
        )

        self.head = Mlp(
            in_features=self.hparams.n_hidden,
            hidden_features=self.hparams.n_hidden,
            out_features=self.hparams.n_out
            * self.hparams.patch_size
            * self.hparams.patch_size,
        )

        if self.hparams.predict_delta_avg or self.hparams.predict_delta_max:
            self.head_avg = Mlp(
                in_features=self.hparams.n_hidden,
                hidden_features=self.hparams.n_hidden,
                out_features=self.hparams.n_out
                * self.hparams.patch_size
                * self.hparams.patch_size,
            )

    @staticmethod
    def add_model_specific_args(
        parent_parser: Optional[Union[argparse.ArgumentParser, list]] = None
    ):
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]

        parser = argparse.ArgumentParser(parents=parent_parser, add_help=False)

        parser.add_argument("--setting", type=str, default="en21x")
        parser.add_argument("--context_length", type=int, default=10)
        parser.add_argument("--target_length", type=int, default=20)
        parser.add_argument("--patch_size", type=int, default=8)
        parser.add_argument("--n_image", type=int, default=8)
        parser.add_argument("--n_weather", type=int, default=24)
        parser.add_argument("--n_hidden", type=int, default=128)
        parser.add_argument("--n_out", type=int, default=1)
        parser.add_argument("--n_heads", type=int, default=8)
        parser.add_argument("--depth", type=int, default=3)
        parser.add_argument("--mlp_ratio", type=float, default=4.0)
        parser.add_argument("--mtm", type=str2bool, default=False)
        parser.add_argument("--leave_n_first", type=int, default=3)
        parser.add_argument("--p_mtm", type=float, default=0.3)
        parser.add_argument("--p_use_mtm", type=float, default=1.0)
        parser.add_argument("--mask_clouds", type=str2bool, default=False)
        parser.add_argument("--use_weather", type=str2bool, default=True)
        parser.add_argument("--predict_delta", type=str2bool, default=False)
        parser.add_argument("--predict_delta0", type=str2bool, default=False)
        parser.add_argument("--predict_delta_avg", type=str2bool, default=False)
        parser.add_argument("--predict_delta_max", type=str2bool, default=False)
        parser.add_argument("--pvt", type=str2bool, default=False)
        parser.add_argument("--pvt_frozen", type=str2bool, default=False)
        parser.add_argument("--add_last_ndvi", type=str2bool, default=False)
        parser.add_argument("--add_mean_ndvi", type=str2bool, default=False)
        parser.add_argument("--spatial_shuffle", type=str2bool, default=False)

        return parser

    def forward(self, data, pred_start: int = 0, preds_length: Optional[int] = None):

        # Input handling

        preds_length = 0 if preds_length is None else preds_length

        c_l = self.hparams.context_length if self.training else pred_start

        hr_dynamic_inputs = data["dynamic"][0]
        hr_dynamic_mask = data["dynamic_mask"][0]

        B, T, C, H, W = hr_dynamic_inputs.shape

        if (
            T == c_l
        ):  # If Only given Context images, add zeros (later overwritten by token mask)
            hr_dynamic_inputs = torch.cat(
                (
                    hr_dynamic_inputs,
                    torch.zeros(
                        B, preds_length, C, H, W, device=hr_dynamic_inputs.device
                    ),
                ),
                dim=1,
            )
            hr_dynamic_mask = torch.cat(
                (
                    hr_dynamic_mask,
                    torch.zeros(
                        B, preds_length, 1, H, W, device=hr_dynamic_mask.device
                    ),
                ),
                dim=1,
            )
            B, T, C, H, W = hr_dynamic_inputs.shape

        static_inputs = data["static"][0][:, :3, ...]

        if self.hparams.spatial_shuffle:
            perm = torch.randperm(B * H * W, device=hr_dynamic_inputs.device)
            invperm = inverse_permutation(perm)

            hr_dynamic_inputs = (
                hr_dynamic_inputs.permute(1, 2, 0, 3, 4)
                .reshape(T, C, B * H * W)[:, :, perm]
                .reshape(T, C, B, H, W)
                .permute(2, 0, 1, 3, 4)
            )

            static_inputs = (
                static_inputs.permute(1, 0, 2, 3)
                .reshape(3, B * H * W)[:, perm]
                .reshape(3, B, H, W)
                .permute(1, 0, 2, 3)
            )

        weather = data["dynamic"][1]
        _, t_m, c_m = weather.shape

        static_inputs = static_inputs.unsqueeze(1).repeat(1, T, 1, 1, 1)

        images = torch.cat([hr_dynamic_inputs, static_inputs], dim=2)
        B, T, C, H, W = images.shape

        # Patchify

        if self.hparams.pvt:
            image_patches_embed = self.embed_images(images)
            B_patch, N_patch, C_patch = image_patches_embed.shape
        else:
            image_patches = (
                images.reshape(
                    B,
                    T,
                    C,
                    H // self.hparams.patch_size,
                    self.hparams.patch_size,
                    W // self.hparams.patch_size,
                    self.hparams.patch_size,
                )
                .permute(0, 3, 5, 1, 2, 4, 6)
                .reshape(
                    B * H // self.hparams.patch_size * W // self.hparams.patch_size,
                    T,
                    C * self.hparams.patch_size * self.hparams.patch_size,
                )
            )
            B_patch, N_patch, C_patch = image_patches.shape
            image_patches_embed = self.embed_images(image_patches)


        mask_patches = (
            hr_dynamic_mask.reshape(
                B,
                T,
                1,
                H // self.hparams.patch_size,
                self.hparams.patch_size,
                W // self.hparams.patch_size,
                self.hparams.patch_size,
            )
            .permute(0, 3, 5, 1, 2, 4, 6)
            .reshape(
                B * H // self.hparams.patch_size * W // self.hparams.patch_size,
                T,
                1 * self.hparams.patch_size * self.hparams.patch_size,
            )
        )

        weather_patches = (
            weather.reshape(B, 1, t_m, c_m)
            .repeat(
                1, H // self.hparams.patch_size * W // self.hparams.patch_size, 1, 1
            )
            .reshape(B_patch, t_m, c_m)
        )

        # Embed Patches

        weather_patches_embed = self.embed_weather(weather_patches)

        # Add Token Mask

        if (
            self.hparams.mtm
            and self.training
            and ((torch.rand(1) <= self.hparams.p_use_mtm).all())
        ):
            token_mask = (
                (
                    torch.rand(B_patch, N_patch, device=weather_patches.device)
                    < self.hparams.p_mtm
                )
                .type_as(weather_patches)
                .reshape(B_patch, N_patch, 1)
                .repeat(1, 1, self.hparams.n_hidden)
            )
            token_mask[:, : self.hparams.leave_n_first] = 0

        else:
            token_mask = (
                torch.ones(B_patch, N_patch, device=weather_patches.device)
                .type_as(weather_patches)
                .reshape(B_patch, N_patch, 1)
                .repeat(1, 1, self.hparams.n_hidden)
            )
            token_mask[:, :c_l] = 0

        image_patches_embed[token_mask.bool()] = (
            (self.mask_token)
            .reshape(1, 1, self.hparams.n_hidden)
            .repeat(B_patch, N_patch, 1)[token_mask.bool()]
        )

        if self.hparams.mask_clouds:
            cloud_mask = (
                (mask_patches.max(-1, keepdim=True)[0] > 0)
                .bool()
                .repeat(1, 1, self.hparams.n_hidden)
            )
            image_patches_embed[cloud_mask] = (
                (self.mask_token)
                .reshape(1, 1, self.hparams.n_hidden)
                .repeat(B_patch, N_patch, 1)[cloud_mask]
            )

        # Add Image and Weather Embeddings
        if self.hparams.use_weather:
            patches_embed = image_patches_embed + weather_patches_embed
        else:
            patches_embed = image_patches_embed

        # Add Positional Embedding
        pos_embed = (
            get_sinusoid_encoding_table(N_patch, self.hparams.n_hidden)
            .to(patches_embed.device)
            .unsqueeze(0)
            .repeat(B_patch, 1, 1)
        )

        x = patches_embed + pos_embed

        # Then feed all into Transformer Encoder
        for blk in self.blocks:
            x = blk(x)

        # Decode image patches
        x_out = self.head(x)

        # Mask Non-masked inputs
        x_out[
            ~token_mask.bool()[
                :,
                :,
                : self.hparams.n_out
                * self.hparams.patch_size
                * self.hparams.patch_size,
            ]
        ] = -1

        # unpatchify images
        images_out = (
            x_out.reshape(
                B,
                H // self.hparams.patch_size,
                W // self.hparams.patch_size,
                N_patch,
                self.hparams.n_out,
                self.hparams.patch_size,
                self.hparams.patch_size,
            )
            .permute(0, 3, 4, 1, 5, 2, 6)
            .reshape(B, N_patch, self.hparams.n_out, H, W)
        )

        if self.hparams.add_last_ndvi:
            mask = hr_dynamic_mask[:, :c_l, ...]

            indxs = (
                torch.arange(c_l, device=mask.device)
                .expand(B, self.hparams.n_out, H, W, -1)
                .permute(0, 4, 1, 2, 3)
            )

            ndvi = hr_dynamic_inputs[:, :c_l, : self.hparams.n_out, ...]

            last_pixel = torch.gather(
                ndvi, 1, (indxs * (mask < 1)).argmax(1, keepdim=True)
            )

            images_out += last_pixel.repeat(1, N_patch, 1, 1, 1)

        elif self.hparams.add_mean_ndvi:
            mask = hr_dynamic_mask[:, :c_l, ...]
            ndvi = hr_dynamic_inputs[:, :c_l, : self.hparams.n_out, ...]

            mean_ndvi = (
                (ndvi * (mask < 1)).sum(1, keepdim=True)
                / ((mask < 1).sum(1, keepdim=True) + 1e-8)
            ).clamp(min=-1.0, max=1.0)

            images_out += mean_ndvi.repeat(1, N_patch, 1, 1, 1)

        if self.hparams.predict_delta_avg:

            image_avg = self.head_avg(x[:, :c_l, :].mean(1).unsqueeze(1))
            image_avg_out = (
                image_avg.reshape(
                    B,
                    H // self.hparams.patch_size,
                    W // self.hparams.patch_size,
                    1,
                    self.hparams.n_out,
                    self.hparams.patch_size,
                    self.hparams.patch_size,
                )
                .permute(0, 3, 4, 1, 5, 2, 6)
                .reshape(B, 1, self.hparams.n_out, H, W)
            )

            images_out += image_avg_out.repeat(1, N_patch, 1, 1, 1)

        elif self.hparams.predict_delta_max:
            image_avg = self.head_avg(x[:, :c_l, :].max(1)[0]).unsqueeze(1)
            image_avg_out = (
                image_avg.reshape(
                    B,
                    H // self.hparams.patch_size,
                    W // self.hparams.patch_size,
                    1,
                    self.hparams.n_out,
                    self.hparams.patch_size,
                    self.hparams.patch_size,
                )
                .permute(0, 3, 4, 1, 5, 2, 6)
                .reshape(B, 1, self.hparams.n_out, H, W)
            )

            images_out += image_avg_out.repeat(1, N_patch, 1, 1, 1)

        elif self.hparams.predict_delta:
            images_out[:, 0, ...] += images[:, 0, : self.hparams.n_out, ...]
            images_out = torch.cumsum(images_out, 1)
        elif self.hparams.predict_delta0:
            images_out += (images[:, :1, : self.hparams.n_out, ...]).repeat(
                1, N_patch, 1, 1, 1
            )

        if not self.training:
            images_out = images_out[:, -preds_length:]

        if self.hparams.spatial_shuffle:
            B, T, C, H, W = images_out.shape
            images_out = (
                images_out.permute(1, 2, 0, 3, 4)
                .reshape(T, C, B * H * W)[:, :, invperm]
                .reshape(T, C, B, H, W)
                .permute(2, 0, 1, 3, 4)
            )

        return images_out, {}
