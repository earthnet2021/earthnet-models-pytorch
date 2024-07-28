from typing import Optional, Union
import argparse
import logging 

import torch
import torch.nn as nn
torch.autograd.set_detect_anomaly(True)

from earthnet_models_pytorch.utils import str2bool

# logging.basicConfig(filename='forward.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class Conv2plus1DBlock(nn.Module):
    """
    Single Conv(2+1)D Block
    """
    def __init__(
        self,
        in_channels: int,
        hid_channels: int, 
        kernel_size: int,
        final_layer: bool = False,
    ):
        super().__init__()

        if final_layer:
            act_fn = nn.Sigmoid()
        else:
            act_fn = nn.ReLU(inplace=True)

        self.block = nn.Sequential(
            # Spatial convolution
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=hid_channels,
                kernel_size=(1, kernel_size, kernel_size), # (1, 3, 3)
                padding='same',
            ),
            nn.BatchNorm3d(hid_channels),
            nn.ReLU(inplace=True),
            # Temporal convolution
            nn.Conv3d(
                in_channels=hid_channels,
                out_channels=hid_channels,
                kernel_size=(kernel_size, 1, 1), # (3, 1, 1)
            ),
            act_fn,
        )

    def forward(self, x):
        return self.block(x)

class StackedConv2plus1D(nn.Module):
    def __init__(self, hparams: argparse.Namespace):
        super().__init__()

        self.hparams = hparams
        kernel_size = self.hparams.kernel_size
        
        # Determine if NDVI prediction is required based on the specified settings
        self.ndvi_pred = (hparams.setting in ["en21-veg", "europe-veg", "en21x", "en22", "en23"])
        
        # Adjust input dimensions based on the selected input type
        if self.hparams.input == 'RGBNR+W':
            input_channels = 49
        elif self.hparams.input == 'NDVI+T+W':
            input_channels = 1 + 1 + 24
        elif self.hparams.input == 'RGBNR':  
            input_channels = 9

        # Define initial block
        initial_block = Conv2plus1DBlock(
            in_channels=input_channels,
            hid_channels=64,
            kernel_size=kernel_size,
        )

        # Define further blocks with similar configuration
        blocks = [initial_block] + [Conv2plus1DBlock(
            in_channels=64,
            hid_channels=64,
            kernel_size=kernel_size,
            ) for _ in range(self.hparams.no_of_blocks)] # if kernel_size=5, no_of_blocks=2; ks=3, n_blocks=3

        # Define network containing upper blocks
        self.net = nn.Sequential(*blocks)

        # Define the final final block
        self.final_block = nn.Sequential(
            Conv2plus1DBlock(
                in_channels=64,
                hid_channels=1,
                kernel_size=kernel_size, # this reduce the len of temporal dimension to 7
                final_layer=True
            )
        )
        
    @staticmethod
    def add_model_specific_args(parent_parser: Optional[Union[argparse.ArgumentParser,list]] = None):
        # TODO remove the useless features
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]
        
        parser = argparse.ArgumentParser(parents = parent_parser, add_help = False)

        parser.add_argument("--bias", type = str2bool, default = True)
        parser.add_argument("--kernel_size", type = int, default = 3)
        parser.add_argument("--no_of_blocks", type = int, default = 3) # Careful with the shape of output
        parser.add_argument("--setting", type = str, default = "en23")
        parser.add_argument("--context_length", type = int, default = 60)
        parser.add_argument("--target_length", type = int, default = 10)
        parser.add_argument("--lc_min", type = int, default = 60)
        parser.add_argument("--lc_max", type = int, default = 90)
        parser.add_argument("--method", type = str, default = None)
        parser.add_argument("--input", type = str, default = "RGBNR")
        parser.add_argument("--target", type = str, default = False)
        parser.add_argument("--residual_connections", type = str2bool, default = False)
        parser.add_argument("--use_weather", type = str2bool, default = True)

        return parser
    
    # Forward pass
    def forward(self, data, pred_start: int = 0, preds_length: Optional[int] = None):

        # Determine the context length for prediction
        context_length = (
            self.hparams.context_length
            if self.training or (pred_start < self.hparams.context_length)
            else pred_start
        )

        # Extract data components

        # sentinel 2 bands
        sentinel = data["dynamic"][0][:, 20 : context_length + 20, ...]

        # Extract the target for the teacher forcing method
        if self.hparams.teacher_forcing and self.training:
            target = data["dynamic"][0][
                :, context_length : context_length + self.hparams.target_length, ...
            ]

        weather = data["dynamic"][1].unsqueeze(3).unsqueeze(4)

        static = data["static"][0]

        # Get the dimensions of the input data. Shape: batch size, temporal size, number of channels, height, width
        b, t_sentinel, c, h, w = sentinel.shape # TODO: remember to change: previously t

        # Expand static tensor along the time dimension
        stacked_static = static.unsqueeze(1).expand(-1, t_sentinel + self.hparams.target_length, -1, -1, -1) # (b, 30, c_s, h, w)

        # Concatenate sentinel and expanded_static along the channel dimension
        stacked_x = torch.cat((sentinel, stacked_static[:, :context_length, ...]), dim=2) # (b, 23, c+c_s, h, w)

        # Prepare input data with weather
        if self.hparams.use_weather:
            weather_array = []
            for t_slice in range(t_sentinel):
                weather_t = (
                    weather[:, t_slice + 20 : t_slice + 5 + 20, ...]
                    .view(weather.shape[0], 1, -1, 1, 1)
                    .squeeze(1)
                    .repeat(1, 1, 128, 128)
                )
                weather_array.append(weather_t.squeeze(1))
            
            stacked_weather = torch.stack(weather_array, dim = 1).contiguous() # (b, 30, c_w, h, w)

            # Concatenate stacked_x and stacked_weather along the channel dimension
            stacked_x = torch.cat(
                (stacked_x, stacked_weather[:, :context_length, ...]), 
                dim=2,
            ).contiguous() # (b, 23, c+c_s+c_w, h, w)

        # Permute the dimensions to match cnn3d input format
        stacked_x = stacked_x.permute(0, 2, 1, 3, 4).contiguous() # (b, c+c_s+c_w, 30, h, w)

        # Forecasting 
        x = self.net(stacked_x)
        x = self.final_block(x)   
            
        preds = x.permute(0, 2, 1, 3, 4).contiguous()
                
        # preds = torch.stack(output, dim = 1).contiguous()
        # logging.debug(f"Output size: {preds.size()}")

        return preds, {}

    