""""Baseline
"""

import torch.nn as nn
import argparse
from typing import Optional, Union
# from pytorch_forecasting import BaseModel
import sys


class Baseline(nn.Module):
    def __init__(self, hparams: argparse.Namespace) -> None:
        super().__init__()
        self.hparams = hparams
        self.linear = nn.Linear(1, 1)

    @staticmethod
    def add_model_specific_args(parent_parser: Optional[Union[argparse.ArgumentParser,list]] = None):
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]
        
        parser = argparse.ArgumentParser(parents = parent_parser, add_help = False)

        parser.add_argument("--context_length", type = int, default = 9)
        parser.add_argument("--target_length", type = int, default = 36)
        return parser

    def forward(self, data, pred_start: int = 0, n_preds: Optional[int] = 0):
        x = self.linear(nn.tensor(1))
        c_l = self.hparams.context_length if self.training else pred_start
        print('here', data["dynamic"][0][:, -1, 0,...].shape)
        output = data["dynamic"][0][:, -1, 0,...].unsqueeze(1).unsqueeze(1).repeat(1, self.hparams.target_length, 1, 1, 1)  
        return output, {}

