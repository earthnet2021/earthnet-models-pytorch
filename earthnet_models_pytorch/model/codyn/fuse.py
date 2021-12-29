


from typing import Optional, Union

import abc

import torch
from torch import nn

from earthnet_models_pytorch.model.codyn.base import MLP, Conv_Block

class BaseFusion(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, extr_enc, extr_skips, extr_mask_enc, extr_mask_skips, stat_enc, stat_skips, stat_mask_enc, stat_mask_skips):
        
        if (stat_enc is not None) and (None in stat_enc):
            idxs = [i for i in range(len(stat_enc)) if stat_enc[i] is None]
            stat_enc = [stat_enc[i] for i in idxs]
            if stat_mask_enc is not None:
                stat_mask_enc = [stat_mask_enc[i] for i in idxs]

        if (stat_mask_enc is not None) and (None in stat_mask_enc):
            ones_base = [e for e in stat_enc if e is not None][0]
            stat_mask_enc = [(m if m is not None else torch.ones_like(ones_base).to(ones_base.device)) for m in stat_mask_enc]

        if (extr_enc is None) and (stat_enc is None):
            enc = None
            mask_enc = None
        elif stat_enc is None:
            enc = extr_enc
            mask_enc = extr_mask_enc
        elif extr_enc is None:
            enc = torch.cat(stat_enc, dim = 2)
            mask_enc = None if stat_mask_enc is None else torch.cat(stat_mask_enc, dim = 2)
        else:
            enc = torch.cat([extr_enc]+stat_enc, dim = 2)
            if (extr_mask_enc is None) and (stat_mask_enc is None):
                mask_enc = None
            elif stat_mask_enc is None:
                mask_enc = torch.cat([extr_mask_enc]+[torch.ones_like(s).to(s.device) for s in stat_enc], dim = 2)
            elif extr_mask_enc is None:
                mask_enc = torch.cat([torch.ones_like(extr_enc).to(extr_enc.device)]+stat_mask_enc, dim = 2)
            else:
                mask_enc = torch.cat([extr_mask_enc]+stat_mask_enc, dim = 2)

        enc = self.fuse_encodings(enc, mask_enc)

        if (stat_skips is not None) and (None in stat_skips):
            idxs = [i for i in range(len(stat_skips)) if stat_skips[i] is None]
            stat_skips = [stat_skips[i] for i in idxs]
            if stat_mask_skips is not None:
                stat_mask_skips = [stat_mask_skips[i] for i in idxs]

        if (stat_mask_skips is not None) and (None in stat_mask_skips):
            ones_base = [s for s in stat_skips if s is not None][0]
            stat_mask_skips = [(m if m is not None else [torch.ones_like(sk).to(sk.device) for sk in ones_base]) for m in stat_mask_skips]

        if (extr_skips is None) and (stat_skips is None):
            skips = None
            mask_skips = None
        elif stat_skips is None:
            skips = extr_skips
            mask_skips = extr_mask_skips
        elif extr_skips is None:
            skips = [torch.cat([s[j] for s in stat_skips], dim = 2) for j in range(len(stat_skips[0]))]
            mask_skips = None if stat_mask_skips is None else [torch.cat([m[j] for m in stat_mask_skips], dim = 2) for j in range(len(stat_mask_skips[0]))]
        else:
            skips = [torch.cat([extr_skips[j]]+[s[j] for s in stat_skips], dim = 2) for j in range(len(stat_skips[0]))]
            if (extr_mask_skips is None) and (stat_mask_skips is None):
                mask_skips = None
            elif stat_mask_skips is None:
                mask_skips = [torch.cat([extr_mask_skips[j]]+[torch.ones_like(s[j]).to(s[j].device) for s in stat_skips], dim = 2) for j in range(len(extr_mask_skips))]
            elif extr_mask_skips is None:
                mask_skips = [torch.cat([torch.ones_like(extr_skips[j]).to(extr_skips[j].device)]+[m[j] for m in stat_mask_skips], dim = 2) for j in range(len(stat_mask_skips[0]))]
            else:
                mask_skips = [torch.cat([extr_mask_skips[j]]+[m[j] for m in stat_mask_skips], dim = 2) for j in range(len(stat_mask_skips[0]))]

        skips = self.fuse_skips(skips, mask_skips)

        return enc, skips

class MLPFusionBlock(nn.Module):

    def __init__(self, enc_input_channels: int, mlp_hidden_channels: int = 64, enc_output_channels: int = 128, n_layers: int = 3, activation: Union[str, list, None] = 'relu'):
        super().__init__()

        self.mlp = MLP(enc_input_channels, mlp_hidden_channels, enc_output_channels, n_layers, activation)

    def forward(self, data, mask):

        if data is None:
            return None
        elif mask is None:
            b, t, c = data.shape
            data = self.mlp(data.reshape(b,t*c))
            return data.reshape(b,t,data.shape[-1])
        else:
            b, t, c = data.shape
            data = self.mlp(data.reshape(b,t*c)*mask.reshape(b,t*c))
            return data.reshape(b,t,data.shape[-1])
            

class ConvFusionBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = None, bias = True, norm = None, activation = None):
        super().__init__()

        self.conv = Conv_Block(in_channels, out_channels, kernel_size, stride, padding, bias, norm, activation)
    
    def forward(self, data, mask):

        if data is None:
            return None
        elif mask is None:
            b, t, c, h, w = data.shape
            data = self.conv(data.reshape(b*t,c,h,w))
            return data.reshape(b,t,*data.shape[1:])
        else:
            b, t, c, h, w = data.shape
            data = self.conv(data.reshape(b*t,c,h,w)*mask.reshape(b*t,c,h,w))
            return data.reshape(b,t,*data.shape[1:])

class ConvSkipFusionBlock(nn.Module):
    def __init__(self, channels, kernel_size = 3, stride = 1, padding = None, bias = True, norm = None, activation = None):
        super().__init__()

        self.conv = nn.ModuleList([ConvFusionBlock(*c, kernel_size, stride, padding, bias, norm, activation) for c in channels])
    
    def forward(self, skips, masks):

        if skips is None:
            return None
        elif masks is None:
            return [self.conv[i](skips[i], None) for i in range(len(skips))]
        else:
            return [self.conv[i](skips[i], masks[i]) for i in range(len(skips))]


class SimpleFusion(BaseFusion):

    def __init__(self, mlp_args: dict, conv_args: dict):
        super().__init__()

        self.fuse_encodings = MLPFusionBlock(**mlp_args)
        self.fuse_skips = ConvSkipFusionBlock(**conv_args)


ALL_FUSION = {"simple": SimpleFusion}

def setup_fusion(setting: dict):
    return ALL_FUSION[setting["name"]](**setting["args"])