"""
Code adapted from https://github.com/gaozhangyang/SimVP-Simpler-yet-Better-Video-Prediction and https://github.com/chengtan9907/SimVPv2 
"""
from typing import Optional, Union

import argparse
import ast

import math

import torch
from torch import nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from earthnet_models_pytorch.utils import str2bool
from earthnet_models_pytorch.model.enc_dec_layer import get_encoder_decoder, ACTIVATIONS

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)  # 1x1
        self.dwconv = DWConv(hidden_features)                  # CFF: Convlutional feed-forward network
        self.act = act_layer()                                 # GELU
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1) # 1x1
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttentionModule(nn.Module):      # Large Kernel Attention
    def __init__(self, dim, kernel_size, dilation=3):
        super().__init__()
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        dd_k = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        dd_p = (dilation * (dd_k - 1) // 2)

        self.conv0 = nn.Conv2d(dim, dim, d_k, padding=d_p, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, dd_k, stride=1, padding=dd_p, groups=dim, dilation=dilation)
        self.conv1 = nn.Conv2d(dim, 2*dim, 1)

        reduction = 16
        self.reduction = max(dim // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // self.reduction, bias=False), # reduction
            nn.ReLU(True),
            nn.Linear(dim // self.reduction, dim, bias=False), # expansion
            nn.Sigmoid()
        )

        # GATE
        self.conv2_0 = nn.Conv2d(dim, dim, d_k, padding=d_p, groups=dim)
        self.conv2_spatial = nn.Conv2d(dim, dim, dd_k, stride=1, padding=dd_p, groups=dim, dilation=dilation)
        self.conv2_1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)           # depth-wise conv
        attn = self.conv_spatial(attn) # depth-wise dilation convolution
        
        f_g = self.conv1(attn)
        split_dim = f_g.shape[1] // 2
        f_x, g_x = torch.split(f_g, split_dim, dim=1)
        return torch.sigmoid(g_x) * f_x


class SpatialAttention(nn.Module):
    def __init__(self, d_model, kernel_size=21):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)         # 1x1 conv
        self.activation = nn.GELU()                          # GELU
        self.spatial_gating_unit = AttentionModule(d_model, kernel_size)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)         # 1x1 conv

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x

class MLP2d(nn.Module):

    def __init__(self, n_in, n_hid, n_out, activation = "relu"):
        super().__init__()

        self.conv1 = nn.Conv2d(n_in, n_hid, 1)
        self.act = ACTIVATIONS[activation]()
        self.conv2 = nn.Conv2d(n_hid, n_out, 1)
    
    def forward(self, x):
        return self.conv2(self.act(self.conv1(x)))


class FiLM(nn.Module):

    def __init__(self, n_c, n_hid, n_x):
        super().__init__()

        self.beta = MLP2d(n_c, n_hid, n_x)
        self.gamma = MLP2d(n_c, n_hid, n_x)

    def forward(self, x, c):
        beta = self.beta(c)
        gamma = self.gamma(c)
        return beta + gamma * x

class FiLMBlock(nn.Module):

    def __init__(self, n_c, n_hid, n_x, activation = "relu"):
        super().__init__()
        self.FiLM = FiLM(n_c, n_hid, n_x)
        self.conv = nn.Conv2d(n_x, n_x, 1)
        self.norm = nn.BatchNorm2d(n_x)
        self.act = ACTIVATIONS[activation]()

    def forward(self, x, c):
        x2 = self.conv(x)
        x2 = self.norm(x2)
        x2 = self.FiLM(x2, c)
        x2 = self.act(x2)
        return x + x2



class GASubBlock(nn.Module):
    def __init__(self, dim, kernel_size=21, mlp_ratio=4., drop=0., drop_path=0.1, act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim, kernel_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x




class GroupConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, act_norm=False):
        super(GroupConv2d, self).__init__()
        self.act_norm = act_norm
        if in_channels % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,groups=groups)
        self.norm = nn.GroupNorm(groups,out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y


class Inception(nn.Module):
    def __init__(self, C_in, C_hid, C_out, incep_ker=[3,5,7,11], groups=8):        
        super(Inception, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)
        layers = []
        for ker in incep_ker:
            layers.append(GroupConv2d(C_hid, C_out, kernel_size=ker, stride=1, padding=ker//2, groups=groups, act_norm=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        return y



class GABlock(nn.Module):
    def __init__(self, in_channels, out_channels, mlp_ratio=8., drop=0.0, drop_path=0.0):
        super(GABlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block = GASubBlock(in_channels, kernel_size=21, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path, act_layer=nn.GELU)

        if in_channels != out_channels:
            self.reduction = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            
    def forward(self, x):
        z = self.block(x)
        return z if self.in_channels == self.out_channels else self.reduction(z)

class Mid_GANet(nn.Module):
    def __init__(self, channel_in, channel_hid, channel_out, N2, mlp_ratio=4., drop=0.0, drop_path=0.1):
        super(Mid_GANet, self).__init__()

        self.N2 = N2
        enc_layers = [GABlock(channel_in, channel_hid, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)]
        for i in range(1, N2-1):
            enc_layers.append(GABlock(channel_hid, channel_hid, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path))
        enc_layers.append(GABlock(channel_hid, channel_out, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path))
        self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        #B, T, C, H, W = x.shape
        #x = x.reshape(B, T*C, H, W)
        B, TC, H, W = x.shape

        z = x
        for i in range(self.N2):
            z = self.enc[i](z)

        #y = z.reshape(B, T, C, H, W)
        return z


class Mid_Xnet(nn.Module):
    def __init__(self, channel_in, channel_hid, channel_out, N_T, incep_ker = [3,5,7,11], groups=8):
        super(Mid_Xnet, self).__init__()

        self.N_T = N_T
        enc_layers = [Inception(channel_in, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))

        dec_layers = [Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_out, incep_ker= incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        #B, T, C, H, W = x.shape
        #x = x.reshape(B, T*C, H, W)
        B, TC, H, W = x.shape

        # encoder
        skips = []
        z = x
        for i in range(self.N_T):
            z = self.enc[i](z)
            if i < self.N_T - 1:
                skips.append(z)

        # decoder
        z = self.dec[0](z)
        for i in range(1, self.N_T):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

        #y = z.reshape(B, T, C, H, W)
        return z

class Mid_Patchnet(nn.Module):
    def __init__(self, channel_in, channel_hid, channel_out, down_factor = 4, filter_size=3, norm = "group", act = "leakyrelu", skip_connection = True, bias = True):
        super().__init__()

        self.enc, self.dec = get_encoder_decoder(
             "PatchMerge", channel_in, channel_hid, channel_out, down_factor = down_factor, filter_size=filter_size, norm=norm, act=act, readout_act=None, skip_connection=True, bias = True
        )

    def forward(self, x):

        x, skips = self.enc(x)
        x = self.dec(x, skips)

        return x

class Mid_ResNet(nn.Module):
    def __init__(self, channel_in, channel_hid, channel_out, n_layers = 2, filter_size = 1, norm = "group", act = "leakyrelu", bias = True):
        super().__init__()

        if norm:
            bias = False

        layers = []
        for i in range(n_layers):

            layer = []

            curr_in_channels = channel_in if i == 0 else channel_hid
            curr_out_channels = channel_out if i == (n_layers - 1) else channel_hid

            layer.append(
                nn.Conv2d(curr_in_channels, curr_out_channels, filter_size, stride = 1, padding = filter_size//2, bias = bias)
            )

            if norm:
                if norm == "group":
                    layer.append(
                        nn.GroupNorm(16, curr_out_channels)
                    )
                elif norm == "batch":
                    layer.append(
                        nn.BatchNorm2d(curr_out_channels)
                    )
                elif norm == "layer":
                    layer.append(
                        nn.LayerNorm(curr_out_channels)
                    )
                else:
                    print(f"Norm {norm} not supported in Mid_1x1Net")
            
            if act:
                layer.append(ACTIVATIONS[act]())

            layers.append(nn.Sequential(*layer))
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        x = self.layers[0](x)

        for i in range(1, len(self.layers)-1):
            x = x + self.layers[i](x)
        
        x = self.layers[-1](x)
        
        return x

class SimVP(nn.Module):
    def __init__(self, hparams):#shape_in, hid_S=16, hid_T=256, N_S=4, N_T=8, incep_ker=[3,5,7,11], groups=8):
        super().__init__()

        self.hparams = hparams

        if self.hparams.gsta_processor:
            self.hparams.proc_type = "gsta"

        self.enc, self.dec = get_encoder_decoder(
             self.hparams.encdec_type, self.hparams.enc_in_channels, self.hparams.encdec_hid_channels, self.hparams.dec_out_channels, down_factor = 2**(self.hparams.encdec_depth // 2), filter_size=self.hparams.encdec_filtersize, norm = self.hparams.encdec_norm, act = self.hparams.encdec_act, readout_act=self.hparams.encdec_readoutact, skip_connection=True, bias = True
        )

        if self.hparams.weather_conditioning == "cat":
            proc_in_channels = self.hparams.encdec_hid_channels * self.hparams.context_length + self.hparams.weather_in_channels * self.hparams.target_length
        else:
            proc_in_channels = self.hparams.encdec_hid_channels * self.hparams.context_length
        proc_out_channels = self.hparams.target_length * self.hparams.encdec_hid_channels

        if self.hparams.weather_conditioning == "FiLM":
            self.FiLM_after_enc = FiLMBlock(self.hparams.weather_in_channels * self.hparams.target_length, self.hparams.proc_hid_channels, proc_in_channels)
            self.FiLM_after_proc = FiLMBlock(self.hparams.weather_in_channels * self.hparams.target_length, self.hparams.proc_hid_channels, proc_out_channels)

        if self.hparams.proc_type == "gsta":
            self.hid = Mid_GANet(proc_in_channels, self.hparams.proc_hid_channels, proc_out_channels, self.hparams.proc_depth)
        elif self.hparams.proc_type == "patch":
            self.hid = Mid_Patchnet(proc_in_channels, self.hparams.proc_hid_channels, proc_out_channels, down_factor = 2**(self.hparams.proc_depth // 2), filter_size = self.hparams.proc_filtersize, norm = self.hparams.encdec_norm, act = self.hparams.encdec_act)
        elif self.hparams.proc_type == "resnet":
            self.hid = Mid_ResNet(proc_in_channels, self.hparams.proc_hid_channels, proc_out_channels, n_layers = self.hparams.proc_depth, filter_size = self.hparams.proc_filtersize, norm = self.hparams.encdec_norm, act = self.hparams.encdec_act)
        else:
            self.hid = Mid_Xnet(proc_in_channels, self.hparams.proc_hid_channels, proc_out_channels, self.hparams.proc_depth, incep_ker=[3,5,7,11], groups=8)#T*hid_S, hid_T, N_T, incep_ker, groups)
    
    @staticmethod
    def add_model_specific_args(parent_parser: Optional[Union[argparse.ArgumentParser,list]] = None):
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]
        
        parser = argparse.ArgumentParser(parents = parent_parser, add_help = False)

        parser.add_argument("--setting", type = str, default = "en21x")
        parser.add_argument("--context_length", type = int, default = 10)
        parser.add_argument("--target_length", type = int, default = 20)
        parser.add_argument("--enc_in_channels", type = int, default = 8)
        parser.add_argument("--encdec_hid_channels", type = int, default = 64)
        parser.add_argument("--encdec_depth", type = int, default = 4)
        parser.add_argument("--weather_in_channels", type = int, default = 24)
        parser.add_argument("--proc_hid_channels", type = int, default = 128)
        parser.add_argument("--proc_depth", type = int, default = 2)
        parser.add_argument("--dec_out_channels", type = int, default = 1)
        parser.add_argument("--gsta_processor", type = str2bool, default = False)
        parser.add_argument("--proc_type", type = str, default = "incep")
        parser.add_argument("--proc_filtersize", type = int, default = 3)
        parser.add_argument("--weather_conditioning", type = str, default = "cat")
        parser.add_argument("--encdec_type", type = str, default = "SimVP")
        parser.add_argument("--encdec_filtersize", type = int, default = 5)
        parser.add_argument("--encdec_norm", type = str, default = None)
        parser.add_argument("--encdec_act", type = str, default = None)
        parser.add_argument("--encdec_readoutact", type = str, default = None)

        return parser

    def forward(self, data, pred_start: int = 0, n_preds: Optional[int] = None):
        
        n_preds = 0 if n_preds is None else n_preds

        c_l = self.hparams.context_length if self.training else pred_start

        hr_dynamic_inputs = data["dynamic"][0][:, :c_l, ...]

        B, T, C, H, W = hr_dynamic_inputs.shape

        static_inputs = data["static"][0][:, :3, ...].unsqueeze(1).repeat(1, T, 1, 1, 1)



        x = torch.cat([hr_dynamic_inputs, static_inputs], dim = 2).reshape(B*T, self.hparams.enc_in_channels, H, W)

        embed, skips = self.enc(x)
        _, C_, H_, W_ = embed.shape

        meso_dynamic_inputs = data["dynamic"][1][:,c_l:,...]

        if len(meso_dynamic_inputs.shape) == 3:

            _, t_m, c_m = meso_dynamic_inputs.shape

            meso_dynamic_inputs = meso_dynamic_inputs.reshape(B, t_m, c_m, 1, 1).repeat(1, 1, 1, H_, W_)

        else:
            _, t_m, c_m, h_m, w_m = meso_dynamic_inputs.shape

            meso_dynamic_inputs = meso_dynamic_inputs.reshape(B, t_m//5, 5, c_m, h_m, w_m).mean(2)[:,:,:,39:41,39:41].repeat(1, 1, 1, H_, W_)

        if self.hparams.weather_conditioning == "cat":
            z = torch.cat([embed.reshape(B, T, C_, H_, W_).reshape(B, T*C_, H_, W_), meso_dynamic_inputs.reshape(B, self.hparams.target_length * c_m, H_, W_)], dim = 1)
        elif self.hparams.weather_conditioning == "FiLM":
            z = self.FiLM_after_enc(embed.reshape(B, T, C_, H_, W_).reshape(B, T*C_, H_, W_), meso_dynamic_inputs.reshape(B, self.hparams.target_length * c_m, H_, W_))
        else:
            z = embed.reshape(B, T, C_, H_, W_).reshape(B, T*C_, H_, W_)

        hid = self.hid(z)

        if self.hparams.weather_conditioning == "FiLM":
            hid = self.FiLM_after_proc(hid, meso_dynamic_inputs.reshape(B, self.hparams.target_length * c_m, H_, W_))

        hid = hid.reshape(B, self.hparams.target_length, self.hparams.encdec_hid_channels, H_, W_).reshape(B*self.hparams.target_length, self.hparams.encdec_hid_channels, H_, W_)

        # skip = skip.reshape(B, T, C_, H, W).mean(1).unsqueeze(1).repeat(1, self.hparams.target_length, 1, 1, 1).reshape(B*self.hparams.target_length, C_, H, W)
        skips = self.reshape_skips(skips)
        Y = self.dec(hid, skips)
        Y = Y.reshape(B, self.hparams.target_length, self.hparams.dec_out_channels, H, W)


        return Y, {}
    
    def reshape_skips(self, skips):

        if isinstance(skips, list):
            new_skips = []
            for skip in skips:
                new_skips.append(self.reshape_skip(skip))
            return new_skips

        else:
            return self.reshape_skip(skips)


    def reshape_skip(self, skip):

        BT, C, H, W = skip.shape
        T = self.hparams.context_length
        T_ = self.hparams.target_length
        B = BT//T
        
        return skip.reshape(B, T, C, H, W).mean(1).unsqueeze(1).repeat(1, T_, 1, 1, 1).reshape(B*T_, C, H, W)
