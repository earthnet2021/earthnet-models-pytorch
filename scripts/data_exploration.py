#!/usr/bin/env python3
"""Data exploration script Script
"""

from argparse import ArgumentParser
import os
import sys
import yaml
import numpy as np
import torch
import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import DataLoader
import xarray as xr

from earthnet_models_pytorch.setting import DATASETS #, get_mean_and_std, nan_value
from earthnet_models_pytorch.utils import parse_setting

def get_mean_and_std(dataloader: DataLoader):
    channels_sum, channels_squared_sum, static_sum, static_squared_sum, scalars_sum, scalars_squared_sum, num_batches = 0, 0, 0, 0, 0, 0, 0
    for data in dataloader:
        # Images
        hr = data['dynamic'][0]
        static = data['static'][0]
        # Scalars
        meteo = data['dynamic'][1]

        # Mean over batch, time, height and width, but not over the channels
        # Images
        channels_sum += torch.mean(hr, dim=[0, 1, 3, 4], dtype=torch.float32)
        channels_squared_sum += torch.mean(hr**2, dim=[0, 1, 3, 4], dtype=torch.float32)
        
        static_sum += torch.mean(static, dim=[0, 2, 3], dtype=torch.float64)
        static_squared_sum += torch.mean(static**2, dim=[0, 2, 3], dtype=torch.float32)
        # Scalars
        scalars_sum += torch.mean(meteo, dim=[0, 1], dtype=torch.float64)
        scalars_squared_sum += torch.mean(meteo**2, dim=[0, 1], dtype=torch.float32)
        
        num_batches += 1

        if num_batches % 100 == 0:
            print(num_batches, channels_sum / num_batches, static_sum / num_batches)
    
    mean_images = channels_sum / num_batches
    std_images = (channels_squared_sum / num_batches - mean_images ** 2) ** 0.5
    
    mean_static = static_sum / num_batches
    std_static = (static_squared_sum / num_batches - mean_static ** 2) ** 0.5

    mean_scalars = scalars_sum / num_batches
    std_scalars = (scalars_squared_sum / num_batches - mean_scalars ** 2) ** 0.5

    return (mean_images, std_images), (mean_static, std_static), (mean_scalars, std_scalars)

    
def nan_value(dataloader: DataLoader):
    meteo_nan, total_meteo = torch.zeros(33), torch.zeros(33)
    static_nan, total_static, num_batches = 0, 0, 0

    for data in dataloader:
        static = data['static'][0]
        meteo = data['dynamic'][1]

        static_is_nan = torch.sum(torch.isnan(static))
        if static_is_nan:
            static_nan += 1
            total_static += static_is_nan
        if torch.sum(torch.isnan(meteo)):
            for b in range(4):
                meteo_is_nan = torch.sum(torch.isnan(meteo[b,...]), dim=0)
                meteo_nan += meteo_is_nan
                print(meteo_is_nan)
                total_meteo += torch.where(meteo_is_nan == 0, meteo_is_nan, 1)
        
        if num_batches % 100 == 0:
            print(num_batches, static_nan, total_meteo)

        num_batches += 1

    return static_nan, meteo_nan


def negative_ndvi(dataloader: DataLoader):
    num_batches, total_neg = 0, 0
    min_lc = 2
    max_lc = 6
    targ_cube["ndvi_target"] = (targ_cube.nir - targ_cube.red)/(targ_cube.nir+targ_cube.red+1e-6)
    
    for data in dataloader:
        targ_path = Path(data["filepath"][j])
        targ_cube = xr.open_dataset(targ_path)
        targ_cube["ndvi_target"] = (targ_cube.nir - targ_cube.red)/(targ_cube.nir+targ_cube.red+1e-6)

        ndvi = data['dynamic'][0][:,:,0,...]
        lc = data["landcover"]

        masks = ((lc >= min_lc).bool() & (lc <= max_lc).bool()).type_as(ndvi).repeat(1, 45, 1, 1)

        masks = torch.where(masks.bool(), (ndvi >= 0).type_as(masks), masks)

        ndvi = ndvi * masks

        neg_ndvi = torch.sum(ndvi < 0, dim=[1,2,3])
        print(neg_ndvi)
        total_neg += torch.sum(ndvi < 0)
        
        #if num_batches % 100 == 0:
        num_batches += 1

    return total_neg/(4*num_batches)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('setting', type = str, metavar='path/to/setting.yaml', help='yaml with all settings')

    args = parser.parse_args()

    # Disabling PyTorch Lightning automatic SLURM detection
    for k, v in os.environ.items():
        if k.startswith("SLURM"):
            del os.environ[k]

    setting_dict = parse_setting(args.setting)
    '''
    setting_file = Path(args.setting)
    with open(setting_file, 'r') as fp:
        setting_dict = yaml.load(fp, Loader = yaml.FullLoader)

    print(setting_dict)'''
    pl.seed_everything(setting_dict["Seed"])
    # Data
    data_args = ["--{}={}".format(key,value) for key, value in setting_dict["Data"].items()]
    data_parser = ArgumentParser()
    data_parser = DATASETS[setting_dict["Setting"]].add_data_specific_args(data_parser)
    data_params = data_parser.parse_args(data_args)
    dm = DATASETS[setting_dict["Setting"]](data_params)

    print(negative_ndvi(dm.train_dataloader()))
    sys.exit()
    if setting_dict["Function"] == 'nan_value':
        print(nan_value(dm.train_dataloader()))
    elif setting_dict["Function"] == 'mean_and_std':
        print(get_mean_and_std(dm.train_dataloader()))
    elif setting_dict["Function"] == 'neg_ndvi':
        print(negative_ndvi(dm.train_dataloader()))

        