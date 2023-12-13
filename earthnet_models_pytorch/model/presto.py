from typing import Optional, Union
import argparse
import torch.nn as nn
import torch
from earthnet_models_pytorch.utils import str2bool

import copy
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm
import sys
sys.path.insert(1, '/Net/Groups/BGI/scratch/crobin/PythonProjects/presto/')
import presto
import earthnet_models_pytorch.model.presto_processing_data as presto_processing_data
import torch

# this is to silence the xarray deprecation warning.
# Our version of xarray is pinned, but we'll need to fix this
# when we upgrade
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


class Presto(nn.Module):
    def __init__(self, hparams: argparse.Namespace):
        super().__init__()

        self.hparams = hparams
        self.encoder_decoder = presto.Presto.load_pretrained()

    @staticmethod
    def add_model_specific_args(
        parent_parser: Optional[Union[argparse.ArgumentParser, list]] = None
    ):
        """
        Add model-specific arguments to the command-line argument parser.

        Parameters
        ----------
        parent_parser: Optional[Union[argparse.ArgumentParser, list]]
            Parent argument parser (optional).

        Returns
        -------
        argparse.ArgumentParser
            Argument parser with added model-specific arguments.
        """
        # Create a new argument parser or use the parent parser
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]

        parser = argparse.ArgumentParser(parents=parent_parser, add_help=False)
        # Add model-specific arguments
        parser.add_argument("--context_length", type=int, default=10)
        parser.add_argument("--target_length", type=int, default=20)
        return parser

    

    def forward(
        self,
        data,
        pred_start: int = 0,
        preds_length: Optional[int] = None,
        step: Optional[int] = None,
    ):
        context_length = (
            self.hparams.context_length
            if self.training or (pred_start < self.hparams.context_length)
            else pred_start
        )
        target_length = self.hparams.target_length

        # Get the dimensions of the input data. Shape: batch size, temporal size, number of channels, height, width
        b, _, _, h, w = data["dynamic"][0][:, :context_length, ...].shape
        
        prediction = torch.zeros([b, target_length, h, w],  device=data["dynamic"][0].device)

        data = presto_processing_data.process_images(data, context_length, target_length)
        month = torch.tensor([0] * data[0].shape[0], device=data[0].device).long() # 


        batch_size = 64

        dl = DataLoader(
            TensorDataset(
                data[0].float(),  # x
                data[1].bool(),  # mask
                data[2].long(),  # dynamic world
                data[3].float(),  # latlons
                data[4].long(),  # pixel id
                month,
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        
        
        for (x, mask, dw, latlons, pixel_id, month) in tqdm(dl):
            # mask target period
            mask[:,context_length:,2:12] = True # s2 bands
            mask[:, context_length:, -1] = True # NDVI
            reconstructed_x, _ = self.encoder_decoder(x[:,0:20,:], dynamic_world=dw[:,0:20], mask=mask[:,0:20,:], latlons=latlons, month=month)

            target = reconstructed_x[:,context_length:20,:]

            # use the period prediction as input
            x_new = copy.deepcopy(x)
            x_new[:, context_length:20,:] = target
            mask[:,context_length:20,2:12] = False # s2 bands
            mask[:,context_length:20, -1] = False # NDVI

            # predict the next 10 timesteop
            reconstructed_x, _ = self.encoder_decoder(x_new[:,context_length:30,:], dynamic_world=dw[:,10:30], mask=mask[:,10:30,:], latlons=latlons, month=month)
            # Concatenate the 2 prediction period
            # prediction[pixel_id[:, 0],:, pixel_id[:, 1], pixel_id[:, 2]] = torch.cat([target[:,:,-1], reconstructed_x[:,context_length:,-1]], dim=1)
            # nir = torch.cat([target[:,:,10], reconstructed_x[:,context_length:,10]], dim = 1)
            # red = torch.cat([target[:,:,5], reconstructed_x[:,context_length:,5]], dim = 1)
            # ndvi = (nir - red) / (nir + red + 1e-8)
            breakpoint()
            print(x[:,:,-1])
        return prediction.unsqueeze(2), {}    


