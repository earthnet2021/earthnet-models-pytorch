
from typing import Optional, Union

import abc

import torch
from torch import nn

from earthnet_models_pytorch.model.codyn.base import Shapes

class BaseExtraction(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, enc: Optional[list] = None, skips: Optional[list] = None, mask_enc: Optional[list] = None, mask_skips: Optional[list] = None):
 
        if mask_skips is not None and all([(m is None) for m in mask_skips]):
            mask_skips = None

        if None in skips:
            idxs = [i for i in range(len(skips)) if skips[i] is None]
            skips = [skips[i] for i in idxs]
            if mask_skips is not None:
                mask_skips = [mask_skips[i] for i in idxs]

        skips = None if (skips is None or len(skips) < 1) else (skips[0] if len(skips) == 1 else [torch.cat([s[j] for s in skips], dim = 2) for j in range(len(skips[0]))])

        if (mask_skips is not None) and (None in mask_skips):
            ones_base = [m for m in mask_skips if m is not None][0]
            mask_skips = [(m if m is not None else [torch.ones_like(sk).to(sk.device) for sk in ones_base]) for m in mask_skips]

        mask_skips = None if (mask_skips is None or skips is None or len(mask_skips) < 1) else ( mask_skips[0] if len(mask_skips) == 1 else [torch.cat([m[j] for m in mask_skips], dim = 2) for j in range(len(mask_skips[0]))])
        skips, mask_skips = self.aggregate_skips(skips, mask_skips)

        if None in enc:
            idxs = [i for i in range(len(enc)) if enc[i] is None]
            enc = [enc[i] for i in idxs]
            if mask_enc is not None:
                mask_enc = [mask_enc[i] for i in idxs]


        enc = None if (enc is None or len(enc) < 1) else (enc[0] if len(enc) == 1 else torch.cat(enc, dim = 2))

        if mask_enc is not None and all([(m is None) for m in mask_enc]):
            mask_enc = None

        if (mask_enc is not None) and (None in mask_enc):
            ones_base = [m for m in mask_enc if m is not None][0]
            mask_enc = [(m if m is not None else torch.ones_like(ones_base).to(ones_base.device)) for m in mask_enc]

        mask_enc = None if (mask_enc is None or enc is None or len(mask_enc) < 1) else (mask_enc[0] if len(mask_enc) == 1 else torch.cat(mask_enc, dim = 2))
        enc, mask_enc = self.aggregate_enc(enc, mask_enc)

        return enc, skips, mask_enc, mask_skips


class SumExtractionBlock(nn.Module):

    def __init__(self, context_length):
        super().__init__()
        self.context_length = context_length

    def forward(self, data, mask):
        if data is None:
            return None, None
        else:
            context_length = min(self.context_length, data.shape[1])
            if self.training:
                t, b = data.shape[1], data.shape[0]
                time = torch.stack([torch.randperm(t)[:context_length] for _ in range(b)], 1).to(data.device)
                index = torch.arange(b).repeat(context_length, 1).to(data.device)
                data = data[index.view(-1), time.view(-1),...].view(b, context_length, *data.shape[2:])
                if mask is None:
                    data = data.mean(1).unsqueeze(1)
                else:
                    mask = mask[index.view(-1), time.view(-1),...].view(b, context_length, *mask.shape[2:])
                    data = torch.where(mask.sum(1) == 0, data.mean(1), (data*mask).sum(1) / (mask.sum(1)+1e-8)).unsqueeze(1)
                    mask = mask.max(1).values.unsqueeze(1)
            else:
                data = data[:,-context_length:,...]
                if mask is None:
                    data = data.mean(1).unsqueeze(1)
                else:
                    mask = mask[:,-context_length:,...]
                    data = torch.where(mask.sum(1) == 0, data.mean(1), (data*mask).sum(1) / (mask.sum(1)+1e-8)).unsqueeze(1)
                    mask = mask.max(1).values.unsqueeze(1)
            return data, mask
        
class SkipsumExtractionBlock(nn.Module):
    def __init__(self, context_length):
        super().__init__()
        self.context_length = context_length
        self.extract = SumExtractionBlock(context_length)
    
    def forward(self, skip_list, mask_list):
        if skip_list is None or len(skip_list) < 1:
            return None, None
        elif mask_list is None:
            mask_list = [None for _ in range(len(skip_list))]
        assert(len(skip_list) == len(mask_list))
        out_skips, out_masks = [], []
        for skip, mask in zip(skip_list,mask_list):
            out_skip, out_mask = self.extract(skip, mask)
            out_skips.append(out_skip)
            out_masks.append(out_mask)
        if None in out_skips:
            return None, None
        elif None in out_masks:
            return out_skips, None
        else:
            return out_skips, out_masks
                    

class SumExtraction(BaseExtraction):

    def __init__(self, context_length):
        super().__init__()

        self.context_length = context_length

        self.aggregate_skips = SkipsumExtractionBlock(self.context_length)
        self.aggregate_mask_skips = SkipsumExtractionBlock(self.context_length)
        self.aggregate_enc = SumExtractionBlock(self.context_length)
        self.aggregate_mask_enc = SumExtractionBlock(self.context_length)

ALL_EXTRACTION = {"sum": SumExtraction}

def setup_extraction(setting):
    return ALL_EXTRACTION[setting["name"]](**setting["args"])