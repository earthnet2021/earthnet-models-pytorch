
from typing import Optional, Union

import argparse
import ast

import torch
import torch.nn as nn

import segmentation_models_pytorch as smp

from earthnet_models_pytorch.utils import str2bool
from earthnet_models_pytorch.model.enc_dec_layer import get_encoder_decoder
from earthnet_models_pytorch.model.conditioning_layer import Identity_Conditioning,Cat_Conditioning,Cat_Project_Conditioning,CrossAttention_Conditioning,FiLMBlock
from earthnet_models_pytorch.model.layer_utils import inverse_permutation

class NextFrameUNet(nn.Module):

    def __init__(self, hparams):

        super().__init__()
        
        self.hparams = hparams

        self.n_tokens_c = 8

        enc_in_channels = self.hparams.n_hr_inputs + self.hparams.n_static_inputs

        c_channels =  self.hparams.n_weather_inputs

        if self.hparams.weather_conditioning_loc == "all":
            enc_dec_proc_conditioning = self.hparams.weather_conditioning
            self.early_conditioning = Identity_Conditioning()
        else:
            enc_dec_proc_conditioning = None
            if self.hparams.weather_conditioning_loc == "early":
                x_channels = enc_in_channels
            
                if self.hparams.weather_conditioning == "cat":
                    self.early_conditioning = Cat_Conditioning()#Cat_Project_Conditioning(x_channels, c_channels)
                    enc_in_channels = x_channels + c_channels
                elif self.hparams.weather_conditioning == "FiLM":
                    self.early_conditioning = FiLMBlock(c_channels, self.hparams.hid_channels, x_channels, norm = self.hparams.norm, act = self.hparams.act)
                elif self.hparams.weather_conditioning == "xAttn":
                    self.early_conditioning = CrossAttention_Conditioning(x_channels, c_channels, n_tokens_c=self.n_tokens_c, act = self.hparams.act, hidden_dim = 16, n_heads=16, mlp_after_attn=self.hparams.mlp_after_attn)
                else:
                    self.early_conditioning = Identity_Conditioning()
            else:
                self.early_conditioning = Identity_Conditioning()

        if self.hparams.weather_conditioning_loc == "skip":
            conditioning_layers = []

            for i in range(self.hparams.enc_depth+1):

                if (i < self.hparams.weather_condition_from_depth):
                    conditioning_layers.append(Identity_Conditioning()) 
                elif self.hparams.weather_conditioning == "FiLM":
                    conditioning_layers.append(
                        FiLMBlock(self.hparams.n_weather_inputs, self.hparams.hid_channels//2, self.hparams.hid_channels, norm = self.hparams.norm, act = self.hparams.act)
                    )
                elif self.hparams.weather_conditioning == "late_fusion":
                    conditioning_layers.append(Cat_Conditioning(self.hparams.hid_channels, self.hparams.n_weather_inputs))
                else:
                    conditioning_layers.append(Identity_Conditioning())

            self.conditioning_layers = nn.ModuleList(conditioning_layers)
        elif self.hparams.weather_conditioning_loc == "latent":
            x_channels = self.hparams.hid_channels
            if self.hparams.weather_conditioning == "cat":
                self.latent_conditioning = Cat_Project_Conditioning(x_channels, c_channels)
            elif self.hparams.weather_conditioning == "FiLM":
                self.latent_conditioning = FiLMBlock(c_channels, self.hparams.hid_channels, x_channels, norm = self.hparams.norm, act = self.hparams.act)
            elif self.hparams.weather_conditioning == "xAttn":
                self.latent_conditioning = CrossAttention_Conditioning(x_channels, c_channels, n_tokens_c=self.n_tokens_c, act = self.hparams.act, hidden_dim = 16, n_heads=16, mlp_after_attn=self.hparams.mlp_after_attn)
            else:
                self.latent_conditioning = Identity_Conditioning()

        self.enc, self.dec = get_encoder_decoder("PatchMerge", enc_in_channels, self.hparams.hid_channels, self.hparams.n_hr_inputs, down_factor=2**self.hparams.enc_depth, filter_size = self.hparams.filter_size, skip_connection=True, norm = self.hparams.norm, act = self.hparams.act, bias = self.hparams.bias, readout_act=self.hparams.readout_act,conditioning = enc_dec_proc_conditioning, c_channels_enc = c_channels,c_channels_dec = c_channels, n_tokens_c = self.n_tokens_c)

    @staticmethod
    def add_model_specific_args(parent_parser: Optional[Union[argparse.ArgumentParser,list]] = None):
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]
        
        parser = argparse.ArgumentParser(parents = parent_parser, add_help = False)
        
        parser.add_argument("--setting", type = str, default = "en21-std")
        parser.add_argument("--context_length", type = int, default = 10)
        parser.add_argument("--target_length", type = int, default = 20)
        parser.add_argument("--enc_depth", type = int, default = 5)
        parser.add_argument("--hid_channels", type = int, default = 128)
        parser.add_argument("--filter_size", type = int, default = 3)
        parser.add_argument("--bias", type = str2bool, default = True)
        parser.add_argument("--norm", type = str, default = None)
        parser.add_argument("--act", type = str, default = None)
        parser.add_argument("--readout_act", type = str, default = "tanh")
        parser.add_argument("--n_hr_inputs", type = int, default = 5)
        parser.add_argument("--n_static_inputs", type = int, default = 3)
        parser.add_argument("--n_weather_inputs", type = int, default = 24)
        parser.add_argument("--weather_conditioning", type = str, default = "FiLM")
        parser.add_argument("--weather_conditioning_loc", type = str, default = "skips")
        parser.add_argument("--weather_condition_from_depth", type = int, default = 0)
        parser.add_argument("--mlp_after_attn", type = str2bool, default = False)        
        parser.add_argument("--spatial_shuffle", type = str2bool, default = False)        

        return parser
    
    def forward(self, data, pred_start: int = 0, preds_length: Optional[int] = None):
    
        preds_length = 0 if preds_length is None else preds_length

        c_l = self.hparams.context_length if self.training else pred_start

        hr_dynamic_inputs = data["dynamic"][0]#[:, :c_l, ...]

        B, T, C, H, W = hr_dynamic_inputs.shape

        static_inputs = data["static"][0][:, :self.hparams.n_static_inputs, ...]


        meso_dynamic_inputs = data["dynamic"][1]
        if len(meso_dynamic_inputs.shape) == 3:

            _, t_m, c_m = meso_dynamic_inputs.shape

            meso_dynamic_inputs = meso_dynamic_inputs.reshape(B, t_m, c_m, 1, 1)#.repeat(1, 1, 1, H, W)

        else:
            _, t_m, c_m, h_m, w_m = meso_dynamic_inputs.shape

            meso_dynamic_inputs = meso_dynamic_inputs.reshape(B, t_m//5, 5, c_m, h_m, w_m).mean(2)[:,:,:,39:41,39:41]#.repeat(1, 1, 1, H, W)
            t_m = t_m//5

        if self.hparams.spatial_shuffle:
            perm = torch.randperm(B*H*W, device=hr_dynamic_inputs.device)
            invperm = inverse_permutation(perm)

            meso_dynamic_inputs
            if meso_dynamic_inputs.shape[-1] == 1:
                meso_dynamic_inputs = meso_dynamic_inputs.expand(-1, -1, -1, H, W)
            else:
                meso_dynamic_inputs = nn.functional.interpolate(meso_dynamic_inputs, size = (H, W), mode='nearest-exact')

            hr_dynamic_inputs = hr_dynamic_inputs.permute(1, 2, 0, 3, 4).reshape(T, C, B*H*W)[:, :, perm].reshape(T, C, B, H, W).permute(2, 0, 1, 3, 4)
            meso_dynamic_inputs = meso_dynamic_inputs.permute(1, 2, 0, 3, 4).reshape(t_m, c_m, B*H*W)[:, :, perm].reshape(t_m, c_m, B, H, W).permute(2, 0, 1, 3, 4).contiguous()
            static_inputs = static_inputs.permute(1, 0, 2, 3).reshape(self.hparams.n_static_inputs, B*H*W)[:, perm].reshape(self.hparams.n_static_inputs, B, H, W).permute(1, 0, 2, 3)


        preds = []
        for t in range(self.hparams.context_length + self.hparams.target_length - 1):
            
            if t < c_l:
                x = torch.cat([hr_dynamic_inputs[:, t, ...], static_inputs], dim = 1).contiguous()
            else:
                x = torch.cat([x, static_inputs], dim = 1).contiguous()

            c = meso_dynamic_inputs[:,t+1,...]

            if self.hparams.weather_conditioning_loc == "early":
                x = self.early_conditioning(x, c)

            x, skips = self.enc(x)

            if self.hparams.weather_conditioning_loc == "skip":
                for i in range(len(skips)):
                    _, _, H_, W_ = skips[i].shape
                    skips[i] = self.conditioning_layers[i](skips[i], c)#[...,:H_, :W_])
            elif self.hparams.weather_conditioning_loc == "latent":
                x = self.latent_conditioning(x, c)

            x = self.dec(x, skips)

            preds.append(x)

        preds = torch.stack(preds, dim = 1).contiguous()[:, -self.hparams.target_length:, :1, ...]

        if self.hparams.spatial_shuffle:
            preds = preds.permute(1, 2, 0, 3, 4).reshape(self.hparams.target_length, 1, B*H*W)[:, :, invperm].reshape(self.hparams.target_length, 1, B, H, W).permute(2, 0, 1, 3, 4)

        return preds, {}