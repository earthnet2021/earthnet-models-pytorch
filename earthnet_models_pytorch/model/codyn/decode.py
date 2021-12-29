


from typing import Optional, Union

import abc

from copy import deepcopy
from functools import partial

import torch
import torchvision

from torch import nn

from earthnet_models_pytorch.model.codyn.base import Conv_Block, Up_Conv_Block, init_weight

class BaseDecoder(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, skips, content, dynamics):

        assert (skips is None and not self.skip) or (self.skip and skips is not None)

        b, t, _ = dynamics.shape

        enc = torch.cat([content.repeat(1,t,1), dynamics], dim = -1)
        b, t, c = enc.shape
        enc = enc.reshape(t*b, c, 1, 1)
        enc = self.first_upconv(enc)
        for idx, stage in enumerate(self.conv):
            if skips is not None:
                skip = skips[idx].repeat(1,t,1,1,1)
                _, _, c, h, w = skip.shape
                skip = skip.reshape(t*b,c,h,w)
                enc = torch.cat([enc, skip], dim = 1)
            enc = stage(enc)
        
        _, c, h, w = enc.shape
        enc = enc.reshape(b, t, c, h, w)
        return enc


class VGG128Decoder(BaseDecoder):
    def __init__(self, output_channels, latent_channels, last_filters, skip = True, init_args: dict = {'init_type': 'normal', 'init_gain': 0.02}):
        super().__init__()

        coef = 2 if skip else 1
        self.skip = skip

        self.first_upconv = nn.Sequential(
            Up_Conv_Block(latent_channels, 12*last_filters, 4, 1, 0, bias = False, norm = "bn", activation = "leaky_relu"),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

        self.conv = nn.ModuleList([
            nn.Sequential(
                Conv_Block(12*last_filters*coef, 12*last_filters, 3, 1, 1, bias = False, norm = "bn", activation = "leaky_relu"),
                Conv_Block(12*last_filters, 12*last_filters, 3, 1, 1, bias = False, norm = "bn", activation = "leaky_relu"),
                Conv_Block(12*last_filters, 6*last_filters, 3, 1, 1, bias = False, norm = "bn", activation = "leaky_relu"),
                nn.Upsample(scale_factor=2, mode='nearest')
            ),
            nn.Sequential(
                Conv_Block(6*last_filters*coef, 6*last_filters, 3, 1, 1, bias = False, norm = "bn", activation = "leaky_relu"),
                Conv_Block(6*last_filters, 6*last_filters, 3, 1, 1, bias = False, norm = "bn", activation = "leaky_relu"),
                Conv_Block(6*last_filters, 4*last_filters, 3, 1, 1, bias = False, norm = "bn", activation = "leaky_relu"),
                nn.Upsample(scale_factor=2, mode='nearest')
            ),
            nn.Sequential(
                Conv_Block(4*last_filters*coef, 2*last_filters, 3, 1, 1, bias = False, norm = "bn", activation = "leaky_relu"),
                Conv_Block(2*last_filters, 2*last_filters, 3, 1, 1, bias = False, norm = "bn", activation = "leaky_relu"),
                Conv_Block(2*last_filters, 2*last_filters, 3, 1, 1, bias = False, norm = "bn", activation = "leaky_relu"),
                nn.Upsample(scale_factor=2, mode='nearest')
            ),
            nn.Sequential(
                Conv_Block(2*last_filters*coef, 2*last_filters, 3, 1, 1, bias = False, norm = "bn", activation = "leaky_relu"),
                Conv_Block(2*last_filters, 2*last_filters, 3, 1, 1, bias = False, norm = "bn", activation = "leaky_relu"),
                Conv_Block(2*last_filters, last_filters, 3, 1, 1, bias = False, norm = "bn", activation = "leaky_relu"),
                nn.Upsample(scale_factor=2, mode='nearest')
            ),
            nn.Sequential(
                Conv_Block(last_filters*coef, 2*last_filters, 3, 1, 1, bias = False, norm = "bn", activation = "leaky_relu"),
                Conv_Block(2*last_filters, 2*last_filters, 3, 1, 1, bias = False, norm = "bn", activation = "leaky_relu"),
                Conv_Block(2*last_filters, last_filters, 3, 1, 1, bias = False, norm = "bn", activation = "leaky_relu"),
                Conv_Block(last_filters, output_channels, 1, 1, 0, bias = False, norm = None, activation = "sigmoid")
            ),
        ])

        init = partial(init_weight, **init_args)
        self.apply(init)

class ResNet18Decoder(BaseDecoder):
    def __init__(self, output_channels, latent_channels, last_filters, skip = True, init_args: dict = {'init_type': 'normal', 'init_gain': 0.02}):
        super().__init__()

        coef = 2 if skip else 1
        self.skip = skip

        # here get rn 18 and just take the layers from it but no stride....
        rn18 = torchvision.models.resnet18(pretrained = False)
        
        rn18.layer4[0].conv1 = nn.Conv2d(latent_channels, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        rn18.layer4[0].downsample = nn.Sequential(nn.Conv2d(latent_channels, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        rn18.layer4[0].conv2 = nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        rn18.layer4[0].bn2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.first_upconv = nn.Sequential(
            nn.Upsample(scale_factor = 4, mode = 'nearest'),
            rn18.layer4[0],
            rn18.layer3[1],
            nn.Upsample(scale_factor = 2, mode = 'nearest')
            )

        
        rn18.layer3[0].conv1 = nn.Conv2d(coef*256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        rn18.layer3[0].downsample = nn.Sequential(nn.Conv2d(coef*256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False), nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        rn18.layer3[0].conv2 = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        rn18.layer3[0].bn2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        

        rn18.layer2[0].conv1 = nn.Conv2d(coef*128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        rn18.layer2[0].downsample = nn.Sequential(nn.Conv2d(coef*128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False), nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        rn18.layer2[0].conv2 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        rn18.layer2[0].bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        layer1_start = deepcopy(rn18.layer2[0])
        layer1_start.conv1 = nn.Conv2d(coef*64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        layer1_start.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        layer1_start.downsample = nn.Sequential(nn.Conv2d(coef*64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False), nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        layer1_start.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        layer1_start.bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        layer0_start = deepcopy(rn18.layer2[0])
        layer0_start.conv1 = nn.Conv2d(64+int(coef==2)*output_channels, last_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        layer0_start.bn1 = nn.BatchNorm2d(last_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        layer0_start.downsample = nn.Sequential(nn.Conv2d(64+int(coef==2)*output_channels, last_filters, kernel_size=(1, 1), stride=(1, 1), bias=False), nn.BatchNorm2d(last_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        layer0_start.conv2 = nn.Conv2d(last_filters, last_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        layer0_start.bn2 = nn.BatchNorm2d(last_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv = nn.ModuleList([
            nn.Sequential(
                rn18.layer3[0],
                rn18.layer2[1],
                nn.Upsample(scale_factor = 2, mode = 'nearest')
            ),
            nn.Sequential(
                rn18.layer2[0],
                rn18.layer1[1],
                nn.Upsample(scale_factor = 2, mode = 'nearest')
            ),
            nn.Sequential(
                layer1_start,
                deepcopy(rn18.layer1[1]),
                nn.Upsample(scale_factor = 2, mode = 'nearest')
            ),
            nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False),
                rn18.bn1,
                rn18.relu,
                nn.Upsample(scale_factor = 2, mode = 'nearest')
            ),
            nn.Sequential(
                layer0_start,
                Conv_Block(last_filters, output_channels, 1, 1, 0, bias = False, norm = None, activation = "sigmoid")
            )
        ])        

        init = partial(init_weight, **init_args)
        self.apply(init)



ALL_DECODERS = {"vgg128": VGG128Decoder, "resnet18": ResNet18Decoder}


def setup_decoder(setting: dict):
    return ALL_DECODERS[setting["name"]](**setting["args"])