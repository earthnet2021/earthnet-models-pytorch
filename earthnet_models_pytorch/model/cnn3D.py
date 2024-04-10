from typing import Optional, Union
import sys
import argparse
import logging 
import ast
import torch.nn as nn
import torch
torch.autograd.set_detect_anomaly(True)
from earthnet_models_pytorch.utils import str2bool

logging.basicConfig(filename='forward.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class StackedConv3D(nn.Module):
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

        # Define convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=input_channels, out_channels=64, kernel_size=kernel_size, stride=self.hparams.stride_size, padding = "same", bias=self.hparams.bias),
            nn.BatchNorm3d(64),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=49, kernel_size=kernel_size, stride=self.hparams.stride_size, padding = "same", bias=self.hparams.bias),
            nn.BatchNorm3d(49),
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=49, out_channels=64, kernel_size=kernel_size, stride=self.hparams.stride_size, padding = "same", bias=self.hparams.bias),
            nn.BatchNorm3d(64),
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=49, kernel_size=kernel_size, stride=self.hparams.stride_size, padding = "same", bias=self.hparams.bias),
            nn.BatchNorm3d(49),
        )

        # Define the final convolutional layer and activation function
        self.final_conv = nn.Sequential(
            nn.Conv3d(in_channels=49, out_channels=1, kernel_size=(5, 3, 3), stride=(3, 1, 1), padding=(0, 1, 1), bias=self.hparams.bias),
        )

        # Final activation function
        self.activation_output = nn.Sigmoid()

        
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
        parser.add_argument("--stride_size", type = int, default = 1)
        parser.add_argument("--setting", type = str, default = "en23")
        parser.add_argument("--context_length", type = int, default = 60)
        parser.add_argument("--target_length", type = int, default = 10)
        parser.add_argument("--lc_min", type = int, default = 60)
        parser.add_argument("--lc_max", type = int, default = 90)
        parser.add_argument("--method", type = str, default = None)
        parser.add_argument("--input", type = str, default = "RGBNR")
        parser.add_argument("--target", type = str, default = False)
        parser.add_argument("--single_frame_prediciton", type = str2bool, default=True)
        parser.add_argument("--residual_connections", type = str2bool, default = False)
        parser.add_argument("--teacher_forcing", type = str2bool, default = False)
        parser.add_argument("--use_weather", type = str2bool, default = True)
        parser.add_argument("--spatial_shuffle", type = str2bool, default = False)

        return parser
    

    def forward(self, data, pred_start: int = 0, preds_length: Optional[int] = None):

        # Determine the context length for prediction
        context_length = (
            self.hparams.context_length
            if self.training or (pred_start < self.hparams.context_length)
            else pred_start
        )

        step = data["global_step"]
        # Calculate teacher forcing coefficient
        if self.hparams.teacher_forcing:
            k = torch.tensor(0.1 * 10 * 12000)

            teacher_forcing_decay = k / (k + torch.exp(step + (120000 / 2) / k))
        else:
            teacher_forcing_decay = 0

        # Extract data components

        # sentinel 2 bands
        sentinel = data["dynamic"][0][:, : context_length, ...]

        # Extract the target for the teacher forcing method
        if self.hparams.teacher_forcing and self.training:
            target = data["dynamic"][0][
                :, context_length : context_length + self.hparams.target_length, ...
            ]

        weather = data["dynamic"][1].unsqueeze(3).unsqueeze(4)

        static = data["static"][0]

        # Get the dimensions of the input data. Shape: batch size, temporal size, number of channels, height, width
        b, t_sentinel, c, h, w = sentinel.shape # TODO: remember to change: previously t


        if self.hparams.spatial_shuffle:
            perm = torch.randperm(b*h*w, device=sentinel.device)
            invperm = inverse_permutation(perm)

            if weather.shape[-1] == 1:
                weather = weather.expand(-1, -1, -1, h, w)
            else:
                weather = nn.functional.interpolate(weather, size = (h,w), mode='nearest-exact')

            sentinel = sentinel.permute(1, 2, 0, 3, 4).reshape(t, c, b*h*w)[:, :, perm].reshape(t, c, b, h, w).permute(2, 0, 1, 3, 4)
            weather = weather.permute(1, 2, 0, 3, 4).reshape(t_w, c_w, b*h*w)[:, :, perm].reshape(t_w, c_w, b, h, w).permute(2, 0, 1, 3, 4).contiguous()
            static = static.permute(1, 0, 2, 3).reshape(3, b*h*w)[:, perm].reshape(3, b, h, w).permute(1, 0, 2, 3)

        # Expand static tensor along the time dimension
        stacked_static = static.unsqueeze(1).expand(-1, t_sentinel + self.hparams.target_length, -1, -1, -1) # (b, 30, c_s, h, w)

        # Concatenate sentinel and expanded_static along the channel dimension
        stacked_x = torch.cat((sentinel, stacked_static[:, :context_length, ...]), dim=2) # (b, 23, c+c_s, h, w)

        # Prepare input data with weather
        if self.hparams.use_weather:
            weather_array = []
            for t_slice in range(t_sentinel):
                weather_t = (
                    weather[:, t_slice : t_slice + 5, ...]
                    .view(weather.shape[0], 1, -1, 1, 1)
                    .squeeze(1)
                    .repeat(1, 1, 128, 128)
                )
                weather_array.append(weather_t.squeeze(1))
            
            stacked_weather = torch.stack(weather_array, dim = 1).contiguous() # (b, 30, c_w, h, w)

            # Concatenate stacked_x and stacked_weather along the channel dimension
            stacked_x = torch.cat((stacked_x, stacked_weather[:, :context_length, ...]), dim=2).contiguous() # (b, 23, c+c_s+c_w, h, w)

        # Permute the dimensions to match cnn3d input format
        stacked_x = stacked_x.permute(0, 2, 1, 3, 4).contiguous() # (b, c+c_s+c_w, 30, h, w)

        if self.hparams.single_frame_prediciton:
            num = self.hparams.target_length # total loops (TODO)
        else:
            num = 1

        output = []

        x = self.conv1(stacked_x)
        x = nn.ReLU()(x)
        x = self.conv2(x)

        # Add residual connection
        if self.hparams.residual_connections:
            x = x + stacked_x

        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.conv4(x)

        # Add residual connection
        if self.hparams.residual_connections:
            x = x + stacked_x

        x = nn.ReLU()(x)
    
        x = self.final_conv(x)        
        # Output
        x = self.activation_output(x)

        # nf = x.squeeze(2)
        # output += [nf.unsqueeze(1)]

        if self.hparams.single_frame_prediciton:
            preds = torch.cat(output, dim=1) 
        else:
            preds = x.permute(0, 2, 1, 3, 4).contiguous()
                
        # preds = torch.stack(output, dim = 1).contiguous()
        logging.debug(f"Output size: {preds.size()}")

        return preds, {}
