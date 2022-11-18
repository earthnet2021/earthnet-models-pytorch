"""Channel-U-Net
"""

from typing import Optional, Union

import argparse
import ast

import torch
import torchvision

import segmentation_models_pytorch as smp

from torch import nn

from earthnet_models_pytorch.utils import str2bool

class ChannelUNet(nn.Module):

    def __init__(self, hparams: argparse.Namespace):
        super().__init__()

        self.hparams = hparams

        self.ndvi_pred = (hparams.setting in ["en21-veg", "europe-veg"])

        if self.hparams.hr_head:
            classes = self.hparams.args["classes"]
            self.hparams.args["classes"] = (5 if self.ndvi_pred else 4 ) * self.hparams.context_length
            self.hparams.args["activation"] = None
            self.head = nn.Sequential(
                nn.Conv2d((5 if self.ndvi_pred else 4 ) * self.hparams.context_length * 2, 64, 1, stride = 1, padding = 0, bias = True),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace = True),
                nn.Conv2d(64, 32, 1, stride = 1, padding = 0, bias = True),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace = True),
                nn.Conv2d(32, classes, 1, stride = 1, padding = 0, bias = True),
                nn.Sigmoid()
            )

        self.unet = getattr(smp, self.hparams.name)(**self.hparams.args)

        if self.hparams.args["encoder_name"].startswith("timm-efficientnet"):
            for name, param in self.unet.named_parameters():
                if name in ["encoder.conv_head.weight", "encoder.bn2.weight","encoder.bn2.bias"]:
                    print(f"Removing grads for redudent layers {name}")
                    param.requires_grad = False
        elif self.hparams.args["encoder_name"].startswith("efficientnet"):
            for name, param in self.unet.named_parameters():
                if name in ["encoder._conv_head.weight","encoder._bn1.weight","encoder._bn1.bias"]:
                    print(f"Removing grads for redudent layers {name}")
                    param.requires_grad = False

        self.upsample = nn.Upsample(size = (128,128))


    @staticmethod
    def add_model_specific_args(parent_parser: Optional[Union[argparse.ArgumentParser,list]] = None):
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]
        
        parser = argparse.ArgumentParser(parents = parent_parser, add_help = False)

        parser.add_argument("--name", type = str, default = "Unet")
        parser.add_argument("--setting", type = str, default = "en21-std")
        parser.add_argument("--args", type = ast.literal_eval, default = '{"encoder_name": "densenet161", "encoder_weights": "imagenet", "in_channels": 191, "classes": 80, "activation": "sigmoid"}')
        parser.add_argument("--context_length", type = int, default = 10)
        parser.add_argument("--target_length", type = int, default = 20)
        parser.add_argument("--noisy_pixel_mask", type = str2bool, default = False)
        parser.add_argument("--predictors", type = str2bool, default = True)
        parser.add_argument("--hr_head", type = str2bool, default = False)

        return parser

    def forward(self, data, pred_start: int = 0, n_preds: Optional[int] = None):
        
        n_preds = 0 if n_preds is None else n_preds


        n_iter = 1 if self.training or n_preds == self.hparams.target_length else n_preds // self.hparams.target_length

        c_l = self.hparams.context_length if self.training else pred_start

        for j in range(n_iter):

            satimgs = data["dynamic"][0][:,(c_l - self.hparams.context_length):c_l,...] if j == 0 else out[:,-self.hparams.context_length:,...]

            b, t, c, h, w = satimgs.shape

            if self.hparams.noisy_pixel_mask and j == 0:
                masks = data["dynamic_mask"][0][:,(c_l - self.hparams.context_length):c_l,...]
                for i in range(b):
                    all_pixels = satimgs[i,...].permute(1,0,2,3)
                    all_pixels = all_pixels[masks[i,...].permute(1,0,2,3) == 1].reshape(c,-1)
                    if all_pixels.nelement() == 0:
                        continue
                    all_pixels = torch.cat(int(1+satimgs[i,...].nelement()/all_pixels.nelement())*[all_pixels], dim = -1)
                    all_pixels = all_pixels[:,torch.randperm(all_pixels.shape[-1], device = satimgs.device)]
                    all_pixels = all_pixels[:,:t*h*w].reshape(c,t,h,w).permute(1,0,2,3)
                    satimgs[i,...] = torch.where(masks[i,...] == 0, all_pixels, satimgs[i,...])
                
            if self.ndvi_pred and j == 0:
                satimgs = torch.cat([((satimgs[:,:,3,...] - satimgs[:,:,2,...])/(satimgs[:,:,3,...] + satimgs[:,:,2,...] + 1e-6)).unsqueeze(2), satimgs], dim = 2)
                c = c+1

            satimgs = satimgs.reshape(b, t*c, h, w)


            if self.hparams.predictors:
                dem = data["static"][0]
                clim = data["dynamic"][1][:,5*((c_l - self.hparams.context_length) + j*self.hparams.target_length):5*(c_l + (j+1)*self.hparams.target_length),:5,...]
                b, t, c, h2, w2 = clim.shape
                clim = clim.reshape(b, t//5, 5, c, h2, w2).mean(2)[:,:,:,39:41,39:41].reshape(b, t//5 * c, 2, 2)

                inputs = torch.cat((satimgs, dem, self.upsample(clim)), dim = 1)
            else:
                inputs = satimgs

            if n_iter == 1:
                out = self.unet(inputs)
                if self.hparams.hr_head:
                    out = self.head(torch.cat([out,satimgs], dim = 1))
                return out.reshape(b, self.hparams.target_length, 1 if self.ndvi_pred else 4, h, w), {}

            else:
                if j == 0:
                    out = self.unet(inputs)
                    if self.hparams.hr_head:
                        out = self.head(torch.cat([out,satimgs], dim = 1))
                    out = out.reshape(b, self.hparams.target_length, 1 if self.ndvi_pred else 4, h, w)
                else:
                    temp_out = self.unet(inputs)
                    if self.hparams.hr_head:
                        temp_out = self.head(torch.cat([temp_out,satimgs], dim = 1))
                    out = torch.cat([out, temp_out.reshape(b, self.hparams.target_length, 1 if self.ndvi_pred else 4, h, w)], dim = 1)
        
        return out, {}
