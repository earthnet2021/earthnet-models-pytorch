"""CNN
"""

from turtle import forward
from typing import Optional, Union, List

import argparse
import ast
from grpc import dynamic_ssl_server_credentials
from pip import main
import torch.nn as nn
import torch 
import sys
from earthnet_models_pytorch.utils import str2bool

class MLP(nn.Module):
    def __init__(self, dim_features):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(dim_features[0], dim_features[1]),
            nn.ReLU(),
            nn.Linear(dim_features[1], dim_features[2]),
            nn.ReLU(), 
            nn.Linear(dim_features[2], dim_features[3]),
            nn.ReLU()
        )

    def forward(self, weather):
        return self.model(weather)


class CNN(nn.Module):

    def __init__(self, hparams: argparse.Namespace):
        super().__init__()

        self.hparams = hparams
        
        self.ndvi_pred = (hparams.setting in ["en21-veg", "europe-veg", "en21x", "en22"])

        
        # input_encoder = 1  # # TODO find a better solution nb of channel 39 = 5 r,b,g, nr, ndvi + 1 dem + 33 weather
        # input_decoder = 1  # 33 weather

        
        dim_features = [24, 24, 10, 1]
        self.MLP = MLP(dim_features)
        input_encoder = 2
        input_decoder = 2
        
        if self.hparams.input == 'NDVI+T':
            input_encoder = 2
            input_decoder = 2
        elif self.hparams.input == 'NDVI+T+W':
            input_encoder = 1 + 1 + 24
            input_decoder = 1 + 1 + 24
        elif self.hparams.input == 'RGBNR':  
            input_encoder = 4
            input_decoder = 4

        padding = self.hparams.kernel_size // 2, self.hparams.kernel_size // 2

        self.conv1 = nn.Conv2d(in_channels=input_encoder,
                            out_channels=64,     
                            kernel_size=self.hparams.kernel_size,
                            padding=padding,
                            bias=self.hparams.bias)
        self.conv2 = nn.Conv2d(in_channels=64,
                            out_channels=64,     
                            kernel_size=self.hparams.kernel_size,
                            padding=padding,
                            bias=self.hparams.bias)
        self.conv3 = nn.Conv2d(in_channels=64,
                            out_channels=64,     
                            kernel_size=self.hparams.kernel_size,
                            padding=padding,
                            bias=self.hparams.bias)
        self.conv4 = nn.Conv2d(in_channels=64,
                            out_channels=64,     
                            kernel_size=self.hparams.kernel_size,
                            padding=padding,
                            bias=self.hparams.bias)

        self.conv = nn.Conv2d(in_channels=64,
                            out_channels=1,     
                            kernel_size=self.hparams.kernel_size,
                            padding=padding,
                            bias=self.hparams.bias)

        self.activation_output = nn.Sigmoid()


        
    @staticmethod
    def add_model_specific_args(parent_parser: Optional[Union[argparse.ArgumentParser,list]] = None):
        # TODO remove the useless features
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]
        
        parser = argparse.ArgumentParser(parents = parent_parser, add_help = False)

        # parser.add_argument("--input_dim", type=int, default=6)
        parser.add_argument("--bias", type=str2bool, default=True)
        parser.add_argument("--kernel_size", type=int, default=3)
        parser.add_argument("--return_all_layers", type=str2bool, default=False)
        parser.add_argument("--setting", type = str, default = "en22")
        parser.add_argument("--context_length", type = int, default = 9)
        parser.add_argument("--target_length", type = int, default = 36)
        parser.add_argument("--lc_min", type = int, default = 82)
        parser.add_argument("--lc_max", type = int, default = 104)
        parser.add_argument("--method", type = str, default = None)
        parser.add_argument("--input", type = str, default = None)
        parser.add_argument("--skip_connections", type = str2bool, default=False)
        parser.add_argument("--add_conv", type = str2bool, default=False)
        return parser
    

    def forward(self, data, pred_start: int = 0, n_preds: Optional[int] = None):

        c_l = self.hparams.context_length if self.training else pred_start

        # Data
        hr_dynamics = data["dynamic"][0][:,(c_l - self.hparams.context_length):c_l,...]
        target = hr_dynamics[:,:,0,...].unsqueeze(2)
        weather = data["dynamic"][1].unsqueeze(3).unsqueeze(4)
        topology = data["static"][0]

        # Shape
        b, t, _, h, w = data["dynamic"][0].shape

        pred = target[:, -1, :, :]
        output = []
        # forecasting network
        for t in range(self.hparams.target_length):

            # Input
            if self.hparams.input =='NDVI+T':
                pred = torch.cat((pred, topology), dim=1)
            elif self.hparams.input == 'NDVI+T+W':
                weather_t = weather[:,c_l + t,...].repeat(1, 1, 128, 128)  
                pred = torch.cat((pred, topology), dim = 1)
                pred = torch.cat((pred, weather_t), dim = 1)

            pred = self.conv1(pred)
            pred = nn.ReLU()(pred)
            pred = self.conv2(pred)
            pred = nn.ReLU()(pred)
            pred = self.conv3(pred)
            pred = nn.ReLU()(pred)
            pred = self.conv4(pred)
            pred = nn.ReLU()(pred)
            pred = self.conv(pred)

            pred = pred + self.MLP(weather[:,c_l + t,...].squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3).repeat(1, 1, 128, 128)

            # Output
            pred = self.activation_output(pred)
            output += [pred]
            
        output = torch.cat(output, dim=1).unsqueeze(2)
        return output, {}
