

"""Encoders

Encoders get Data and Mask of shape b,t,c,h,w or b,c,h,w.
Mask can also be None to not downsample them.
Encoders are pre-defined to output skips or use mask.
At runtime output skips can be overriden as False.
Encoders output Encodings of shape b,t,l
    + Skips (None or [last->first, shape b,t,c,h,w])
    + Mask_enc (None or b,t,l)
    + Mask_skips (None or [last->first, shape b,t,c,h,w])

setup_encoders is the entrypoint for using encoders

"""


from typing import Optional, Union

import abc

from functools import partial

import torch
import torchvision

from torch import nn

from earthnet_models_pytorch.model.codyn.base import Conv_Block, Mask_Block, init_weight

def mask_50_pct(mask):
    tb, c, h, w = mask.shape
    return (mask.sum(-1).sum(-1).sum(-1) > 0.5 * h * w * c).type_as(mask).unsqueeze(-1)

class BaseEncoder(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, data: torch.Tensor, mask: Optional[torch.Tensor] = None, output_skips: bool = True):
        
        use_mask = (mask is not None) and self.use_mask
        output_skips = output_skips and self.output_skips

        if len(data.shape) == 5:
            b, t, c, h, w = data.shape
        elif len(data.shape) == 4:
            t = 1
            b, c, h, w = data.shape
        elif len(data.shape) == 3:
            b, t, c, = data.shape
            h, w = 1, 1
        else: raise NotImplementedError
        
        data = data.reshape(t*b, c, h, w)
        if output_skips: content_skips = []

        if use_mask:
            mask = mask.reshape(t*b, c, h, w)
            if output_skips: mask_skips = []
            m = mask
        for idx, stage in enumerate(self.conv):
            data = stage(data)
            if output_skips: content_skips.append(data.reshape(b,t,*data.shape[1:]))
            if use_mask:
                m = self.mask[idx](m)
                if output_skips: mask_skips.append(m.reshape(b,t,*m.shape[1:]))
        
        data = self.last_conv(data).reshape(b,t,self.latent_channels)
        m = None if not use_mask else self.last_mask(mask).reshape(b,t,1).repeat(1,1,self.latent_channels)
        content_skips = None if not output_skips else content_skips[::-1]
        mask_skips = None if not (output_skips and use_mask) else mask_skips[::-1]
        
        return data, content_skips, m, mask_skips


class VGG128Encoder(BaseEncoder):
    def __init__(self, in_channels, latent_channels, in_filters, output_skips: bool = True, init_args: dict = {'init_type': 'normal', 'init_gain': 0.02}):
        super().__init__()

        self.output_skips = output_skips
        
        self.use_mask = True

        self.latent_channels = latent_channels

        self.conv = nn.ModuleList([
            nn.Sequential(
                Conv_Block(in_channels, in_filters, 3, 1, 1, bias = False, norm = "bn", activation = "leaky_relu"),
                Conv_Block(in_filters, in_filters, 3, 1, 1, bias = False, norm = "bn", activation = "leaky_relu")
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                Conv_Block(in_filters, 2*in_filters, 3, 1, 1, bias = False, norm = "bn", activation = "leaky_relu"),
                Conv_Block(2*in_filters, 2*in_filters, 3, 1, 1, bias = False, norm = "bn", activation = "leaky_relu")
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                Conv_Block(2*in_filters, 4*in_filters, 3, 1, 1, bias = False, norm = "bn", activation = "leaky_relu"),
                Conv_Block(4*in_filters, 4*in_filters, 3, 1, 1, bias = False, norm = "bn", activation = "leaky_relu")
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                Conv_Block(4*in_filters, 6*in_filters, 3, 1, 1, bias = False, norm = "bn", activation = "leaky_relu"),
                Conv_Block(6*in_filters, 6*in_filters, 3, 1, 1, bias = False, norm = "bn", activation = "leaky_relu"),
                Conv_Block(6*in_filters, 6*in_filters, 3, 1, 1, bias = False, norm = "bn", activation = "leaky_relu")
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                Conv_Block(6*in_filters, 12*in_filters, 3, 1, 1, bias = False, norm = "bn", activation = "leaky_relu"),
                Conv_Block(12*in_filters, 12*in_filters, 3, 1, 1, bias = False, norm = "bn", activation = "leaky_relu"),
                Conv_Block(12*in_filters, 12*in_filters, 3, 1, 1, bias = False, norm = "bn", activation = "leaky_relu")
            )
        ])
        self.last_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Conv_Block(12*in_filters, latent_channels, 4, 1, 0, bias = False, norm = "bn", activation = "tanh")
        )

        self.mask = nn.ModuleList([
            Mask_Block(kernel_sizes = [3,3], out_channels = in_filters, strict = True),
            nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                Mask_Block(kernel_sizes = [3,3], out_channels = 2*in_filters, strict = True)
            ),
            nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                Mask_Block(kernel_sizes = [3,3], out_channels = 4*in_filters, strict = True)
            ),
            nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                Mask_Block(kernel_sizes = [3,3,3], out_channels = 6*in_filters, strict = True)
            ),
            nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                Mask_Block(kernel_sizes = [3,3,3], out_channels = 12*in_filters, strict = True)
            )
        ])

        self.last_mask = mask_50_pct

        init = partial(init_weight, **init_args)
        self.apply(init)

class Center4x80Encoder(BaseEncoder):
    def __init__(self, in_channels, latent_channels, in_filters, output_skips = False, init_args: dict = {'init_type': 'normal', 'init_gain': 0.02}):
        super().__init__()

        self.latent_channels = latent_channels

        self.output_skips = output_skips

        self.conv = nn.ModuleList([
            nn.Sequential(
                Conv_Block(in_channels, in_filters, 3, 1, 1, bias = True, norm = "bn", activation = "leaky_relu"),
                Conv_Block(in_filters, in_filters, 4, 2, None, bias = True, norm = "bn", activation = "leaky_relu") #40
            ),
            nn.Sequential(
                Conv_Block(in_filters, 2*in_filters, 3, 1, 1, bias = True, norm = "bn", activation = "leaky_relu"),
                Conv_Block(2*in_filters, 2*in_filters, 4, 2, None, bias = True, norm = "bn", activation = "leaky_relu") #20
            ),
            nn.Sequential(
                Conv_Block(2*in_filters, 4*in_filters, 3, 1, 1, bias = True, norm = "bn", activation = "leaky_relu"),
                Conv_Block(4*in_filters, 4*in_filters, 4, 4, 0, bias = True, norm = "bn", activation = "leaky_relu") #5
            )
        ])
        self.last_conv = Conv_Block(4*in_filters, latent_channels, 5, 1, 0, bias = False, norm = "bn", activation = "tanh")

        self.use_mask = False

        init = partial(init_weight, **init_args)
        self.apply(init)

class EmptyForward(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, data):
        return data

class ResNet18Encoder(BaseEncoder):
    def __init__(self, in_channels, output_skips: bool = True, pretrained: bool = True):
        # 512 latent channels
        super().__init__()

        self.output_skips = output_skips

        self.use_mask = True

        self.latent_channels = 512

        rn18 = torchvision.models.resnet18(pretrained = pretrained)
        if in_channels < 3:
            first_conv = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            first_conv.weight.data = rn18.conv1.weight.data[:,:in_channels,...]
        elif in_channels > 3:
            first_conv = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            first_conv.weight.data = torch.stack(([rn18.conv1.weight.data[:,2,...],rn18.conv1.weight.data[:,1,...],rn18.conv1.weight.data[:,0,...]] + ((in_channels - 3) * [rn18.conv1.weight.data[:,0,...]])), dim = 1)
        else:
            first_conv = rn18.conv1

        self.conv = nn.ModuleList([
            EmptyForward(),
            nn.Sequential(first_conv, rn18.bn1, rn18.relu),
            nn.Sequential(rn18.maxpool, rn18.layer1),
            rn18.layer2,
            rn18.layer3
        ])

        self.last_conv = nn.Sequential(rn18.layer4,rn18.avgpool)
        self.mask = nn.ModuleList([
            EmptyForward(),
            nn.Sequential(Mask_Block(kernel_sizes=[3,3,3], out_channels = 64), nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)),
            nn.Sequential(nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0), Mask_Block(kernel_sizes=[3,3,3,3],out_channels =  64)),
            nn.Sequential(nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0), Mask_Block(kernel_sizes=[3,3,3,3],out_channels =  128)),
            nn.Sequential(nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0),Mask_Block(kernel_sizes=[3,3,3,3],out_channels =  256))
        ])

        self.last_mask = mask_50_pct


class ResNet50Encoder(BaseEncoder):
    def __init__(self, in_channels, output_skips: bool = True, pretrained: bool = True):
        # 2048 latent channels
        super().__init__()

        self.output_skips = output_skips

        self.use_mask = True

        self.latent_channels = 2048

        rn50 = torchvision.models.resnet50(pretrained = pretrained)

        if in_channels < 3:
            first_conv = nn.Conv2d(in_channels, rn50.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            first_conv.weight.data = rn50.conv1.weight.data[:,:in_channels,...]
        if in_channels > 3:
            first_conv = nn.Conv2d(in_channels, rn50.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            first_conv.weight.data = torch.cat([rn50.conv1.weight.data[:,2,...],rn50.conv1.weight.data[:,1,...],rn50.conv1.weight.data[:,0,...]] + (in_channels - 3) * [rn50.conv1.weight.data[:,0,...]], dim = 1)
        else:
            first_conv = rn50.conv1

        self.conv = nn.ModuleList([
            EmptyForward(),
            nn.Sequential(first_conv, rn50.bn1, rn50.relu),
            nn.Sequential(rn50.maxpool, rn50.layer1),
            rn50.layer2,
            rn50.layer3
        ])

        self.last_conv = nn.Sequential(rn50.layer4,rn50.avgpool)

        self.mask = nn.ModuleList([
            EmptyForward(),
            nn.Sequential(Mask_Block(kernel_sizes=[7], out_channels = rn50.inplanes), nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)),
            nn.Sequential(nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0), Mask_Block(kernel_sizes=[3,3,3],out_channels =  256)),
            nn.Sequential(nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0), Mask_Block(kernel_sizes=[3,3,3,3],out_channels =  512)),
            nn.Sequential(nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0),Mask_Block(kernel_sizes=[3,3,3,3,3,3],out_channels =  1024)),
            nn.Sequential(nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0),Mask_Block(kernel_sizes=[3,3,3],out_channels =  2048)),
        ])

        self.last_mask = mask_50_pct

class DownsampleEncoder(BaseEncoder):
    def __init__(self, n_channels = 1, depth = 5, output_skips = True, use_mask = False):

        super().__init__()

        self.output_skips = output_skips

        self.use_mask = True

        self.latent_channels = n_channels

        self.conv = nn.ModuleList([EmptyForward()] + (depth-1) * [nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)])

        self.last_conv = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)

        self.mask = self.conv
        self.last_mask = mask_50_pct

class CutOutForward(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, data):
        tb, c, h, w = data.shape
        return data[:,:,((h//2)-1):((h//2)+1),((w//2)-1):((w//2)+1)].reshape(tb, c*4)

class CutOutEncoder(BaseEncoder):

    def __init__(self, n_channels = 6):
        super().__init__()
        self.use_mask = False
        self.output_skips = False
        self.latent_channels = n_channels * 4

        self.conv = []
        self.last_conv = CutOutForward()

        self.mask = None
        self.last_mask = None

class PixelThroughpass(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, data):
        return data

class PixelEncoder(BaseEncoder):

    def __init__(self, n_channels = 6):
        super().__init__()
        self.use_mask = False
        self.output_skips = False
        self.latent_channels = n_channels

        self.conv = []
        self.last_conv = PixelThroughpass()

        self.mask = None
        self.last_mask = None


ALL_ENCODERS = {"vgg128": VGG128Encoder, '4x80': Center4x80Encoder, 'resnet50': ResNet50Encoder, 'resnet18': ResNet18Encoder, "downsample": DownsampleEncoder, "cutout": CutOutEncoder, "pixel": PixelEncoder}


def setup_encoders(settings: list):
    return [ALL_ENCODERS[s["name"]](**s["args"]) for s in settings]