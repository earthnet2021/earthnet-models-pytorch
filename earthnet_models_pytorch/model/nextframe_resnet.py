
from typing import Optional, Union

import argparse

import torch
import torch.nn as nn

from earthnet_models_pytorch.utils import str2bool
from earthnet_models_pytorch.model.enc_dec_layer import ACTIVATIONS

class MLP2d(nn.Module):

    def __init__(self, n_in, n_hid, n_out, act = "relu"):
        super().__init__()

        self.conv1 = nn.Conv2d(n_in, n_hid, 1)
        self.act = ACTIVATIONS[act]()
        self.conv2 = nn.Conv2d(n_hid, n_out, 1)
    
    def forward(self, x):
        return self.conv2(self.act(self.conv1(x)))


class FiLM(nn.Module):

    def __init__(self, n_c, n_hid, n_x, act = "relu"):
        super().__init__()

        self.beta = MLP2d(n_c, n_hid, n_x, act = act)
        self.gamma = MLP2d(n_c, n_hid, n_x, act = act)

    def forward(self, x, c):
        beta = self.beta(c)
        gamma = self.gamma(c)
        return beta + gamma * x

class FiLMBlock(nn.Module):

    def __init__(self, n_c, n_hid, n_x, norm = "group", act = "leakyrelu"):
        super().__init__()
        self.FiLM = FiLM(n_c, n_hid, n_x, act = act)
        self.conv = nn.Conv2d(n_x, n_x, 1)
        if norm == "group":
            self.norm = nn.GroupNorm(16, n_x)
        elif norm == "batch":
            self.norm = nn.BatchNorm2d(n_x)
        elif norm == "layer":
            self.norm = nn.LayerNorm(n_x)
        else:
            self.norm = nn.Identity()
        self.act = ACTIVATIONS[act]()

    def forward(self, x, c):
        x2 = self.conv(x)
        x2 = self.norm(x2)
        x2 = self.FiLM(x2, c)
        x2 = self.act(x2)
        return x + x2

class Cat_Conditioning(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, c):
        return torch.cat([x, c], dim = 1).contiguous()
    
class Identity_Conditioning(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, c):
        return x

class NextFrameResNet(nn.Module):

    def __init__(self, hparams):#, channel_in, channel_hid, channel_out, n_layers = 2, filter_size = 1, norm = "group", act = "leakyrelu", bias = True, readout_act = "tanh"):
        super().__init__()

        self.hparams = hparams

        if self.hparams.norm:
            bias = False
        else:
            bias = self.hparams.bias

        layers = []
        conditioning_layers = []
        for i in range(self.hparams.n_layers):

            layer = []

            curr_in_channels = self.hparams.n_hr_inputs + self.hparams.n_static_inputs if i == 0 else self.hparams.hid_channels
            #curr_out_channels = channel_out if i == (n_layers - 1) else channel_hid

            if (self.hparams.weather_conditioning == "earlycat") and (i == 0):
                curr_in_channels = curr_in_channels + self.hparams.n_weather_inputs
                conditioning_layers.append(
                    Cat_Conditioning()
                )
            elif (i > 0) and (self.hparams.weather_conditioning == "FiLM"):
                conditioning_layers.append(
                    FiLMBlock(self.hparams.n_weather_inputs, self.hparams.hid_channels//2, self.hparams.hid_channels, norm = self.hparams.norm, act = self.hparams.act)
                )
            else:
                conditioning_layers.append(Identity_Conditioning())


            layer.append(
                nn.Conv2d(curr_in_channels, self.hparams.hid_channels, self.hparams.filter_size, stride = 1, padding = self.hparams.filter_size//2, bias = bias)
            )

            if self.hparams.norm:
                if self.hparams.norm == "group":
                    layer.append(
                        nn.GroupNorm(16, self.hparams.hid_channels)
                    )
                elif self.hparams.norm == "batch":
                    layer.append(
                        nn.BatchNorm2d(self.hparams.hid_channels)
                    )
                elif self.hparams.norm == "layer":
                    layer.append(
                        nn.LayerNorm(self.hparams.hid_channels)
                    )
                else:
                    print(f"Norm {self.hparams.norm} not supported in NextFrameResNet")
            
            if self.hparams.act:
                layer.append(ACTIVATIONS[self.hparams.act]())

            layers.append(nn.Sequential(*layer))
        
        self.conditioning_layers = nn.ModuleList(conditioning_layers)
        self.layers = nn.ModuleList(layers)

        self.readout = nn.Conv2d(self.hparams.hid_channels, self.hparams.n_hr_inputs, 1, stride = 1, padding = 0, bias = False)

        if self.hparams.readout_act:
            self.readout_act = ACTIVATIONS[self.hparams.readout_act]()
        else:
            self.readout_act = nn.Identity()

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
        parser.add_argument("--n_hr_inputs", type = int, default = 5)
        parser.add_argument("--n_static_inputs", type = int, default = 3)
        parser.add_argument("--n_weather_inputs", type = int, default = 24)
        parser.add_argument("--weather_conditioning", type = str, default = "FiLM")
        parser.add_argument("--hid_channels", type = int, default = 128)
        parser.add_argument("--filter_size", type = int, default = 3)
        parser.add_argument("--bias", type = str2bool, default = True)
        parser.add_argument("--norm", type = str, default = None)
        parser.add_argument("--act", type = str, default = None)
        parser.add_argument("--readout_act", type = str, default = None)
        parser.add_argument("--n_layers", type = int, default = 4)

        return parser

    def forward(self, data, pred_start: int = 0, preds_length: Optional[int] = None):
        
        preds_length = 0 if preds_length is None else preds_length

        c_l = self.hparams.context_length if self.training else pred_start

        hr_dynamic_inputs = data["dynamic"][0]#[:, :c_l, ...]

        B, T, C, H, W = hr_dynamic_inputs.shape

        static_inputs = data["static"][0][:, :3, ...]

        meso_dynamic_inputs = data["dynamic"][1]
        if len(meso_dynamic_inputs.shape) == 3:

            _, t_m, c_m = meso_dynamic_inputs.shape

            meso_dynamic_inputs = meso_dynamic_inputs.reshape(B, t_m, c_m, 1, 1).repeat(1, 1, 1, H, W)

        else:
            _, t_m, c_m, h_m, w_m = meso_dynamic_inputs.shape

            meso_dynamic_inputs = meso_dynamic_inputs.reshape(B, t_m//5, 5, c_m, h_m, w_m).mean(2)[:,:,:,39:41,39:41].repeat(1, 1, 1, H, W)

        preds = []
        for t in range(self.hparams.context_length + self.hparams.target_length - 1):
            
            if t < c_l:
                x = torch.cat([hr_dynamic_inputs[:, t, ...], static_inputs], dim = 1).contiguous()
            else:
                x = torch.cat([x, static_inputs], dim = 1).contiguous()

            c = meso_dynamic_inputs[:,t+1,...]

            x = self.conditioning_layers[0](x, c)
            x = self.layers[0](x)
            for i in range(1,len(self.layers)):
                x = self.conditioning_layers[i](x, c)
                x = x + self.layers[i](x)
            
            x = self.readout(x)
            x = self.readout_act(x)

            preds.append(x)

        preds = torch.stack(preds, dim = 1).contiguous()[:, -self.hparams.target_length:, :1, ...]

        return preds, {}
