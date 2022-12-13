
from typing import Union, Optional

import argparse
import copy
import multiprocessing
import re
import sys

from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr

from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import transforms

from earthnet_models_pytorch.utils import str2bool

variables = {
            # Sentinel 2 bands. Spatio-temporal data. Every 5 days.
            's2_bands': ['s2_B02', 's2_B03', 's2_B04', 's2_B05', 's2_B06', 's2_B07', 's2_B8A'],
            's2_avail': ['s2_avail'],
            's2_scene_classification': ['s2_SCL'],
            'cloud_mask': ['s2_mask'],
            # Sentinel 1 bands. Spatio-temporal data. Every 12 days. 
            's1_bands': ['s1_vv', 's1_vh'],
            's1_avail': ['s1_avail'],
            # Monthly mean and std computed over 1984-2020 using Landsat 5, 7, 8. Digital Earth Africa product. 
            'ndviclim': ['ndviclim_mean', 'ndviclim_std'],
            # Era5 reanalysis dataset. Era5 era5land pev and pet have a low precision
            'era5lands': ['era5land_t2m_mean', 'era5land_pev_mean', 'era5land_slhf_mean', 'era5land_ssr_mean', 'era5land_sp_mean',  'era5land_sshf_mean', 'era5land_e_mean', 'era5land_tp_mean', 
                        'era5land_t2m_min', 'era5land_pev_min', 'era5land_slhf_min', 'era5land_ssr_min', 'era5land_sp_min', 'era5land_sshf_min', 'era5land_e_min', 'era5land_tp_min', 
                        'era5land_t2m_max', 'era5land_pev_max', 'era5land_slhf_max', 'era5land_ssr_max', 'era5land_sp_max', 'era5land_sshf_max', 'era5land_e_max', 'era5land_tp_max'],
            # Era5 recomputed by Fabian Gans for the XAIDA project. More accurate. 
            'era5': ['era5_e', 'era5_pet', 'era5_pev', 'era5_ssrd', 'era5_t2m', 'era5_t2mmax', 'era5_t2mmin', 'era5_tp'],
            # soigrids is not yet define on 128 x 128 pixels. Can maybe not be used during the prediction period.
            'soilgrids': ['sg_bdod_top_mean', 'sg_bdod_sub_mean','sg_cec_top_mean','sg_cec_sub_mean','sg_cfvo_top_mean','sg_cfvo_sub_mean',
                        'sg_clay_top_mean','sg_clay_sub_mean','sg_nitrogen_top_mean','sg_nitrogen_sub_mean','sg_phh2o_top_mean','sg_phh2o_sub_mean',
                        'sg_ocd_top_mean','sg_ocd_sub_mean','sg_sand_top_mean','sg_sand_sub_mean','sg_silt_top_mean','sg_silt_sub_mean','sg_soc_top_mean',
                        'sg_soc_sub_mean'],
            # Elevation map. Defined on 0 - 2000. 
            'elevation': ['cop_dem'], #['srtm_dem', 'alos_dem', 'cop_dem'],
            # Landscape classes. Categorical variable.
            'landcover': ['esawc_lc'],
            # Geomorphon classes. Categorical variable. geomorphon is not define on 128 x 128 pixels neither.
            'geomorphon': ['geom_cls']
            }

class EarthNet2023Dataset(Dataset):

    def __init__(self, folder: Union[Path, str], target: str, variables=variables, fp16 = False):
        if not isinstance(folder, Path):
            folder = Path(folder)
        self.filepaths = sorted(list(folder.glob("*/*.nc"))) # why sorted?
        self.type = np.float16 if fp16 else np.float32
        self.target = target
        self.variables = variables


    def __getitem__(self, idx: int) -> dict:
        
        filepath = self.filepaths[idx]
        minicube = xr.open_dataset(filepath)

        # Select the days with available data
        s2_avail = minicube[self.variables['s2_avail']] 
        index_avail = np.where(s2_avail == 1)[0]

        # Detect the missing day
        missing = index_avail[1:] - index_avail[:-1] - 5
        index_missing = np.where(missing > 0)[0]

        # Create the minicube
        # s2 is 5 days, and already rescaled [0, 1]
        s2_cube = minicube[self.variables['s2_bands']].isel(time=index_avail).to_array()
        s2_cube[np.isnan(s2_cube)] = np.mean(s2_cube)

        s2_mask = minicube[self.variables['cloud_mask']].isel(time=index_avail).to_array()

        # weather is daily
        meteo_cube = minicube[self.variables['era5']].to_array()
        meteo_cube[np.isnan(meteo_cube['era5_tp'])] = 0
        meteo_cube[np.isnan(meteo_cube)] =  np.mean(meteo_cube)
        # TODO normalization

        topography = minicube[self.variables['elevation']].to_array() / 2000 # c h w       
        topography[np.isnan(topography)] = np.mean(topography)

        landcover = minicube[self.variables['landcover']].to_array() # c h w
        # TODO categoritcal variables ?

        target = self.target_computation(minicube)
    
        
        # Final minicube
        data = {
            "dynamic": [
                torch.from_numpy(s2_cube),
                torch.from_numpy(meteo_cube)
            ],
            "dynamic_mask": [torch.from_numpy(s2_mask)],
            "static": [
                torch.from_numpy(topography)
            ],
            "target": torch.from_numpy(target),
            "s2_missing": torch.from_numpy(index_missing),
            "landcover": torch.from_numpy(landcover),
            "filepath": str(filepath),
            "cubename": self.__name_getter(filepath)
        }
        return data

    def __len__(self) -> int:
        return len(self.filepaths)

    def __name_getter(self, path: Path) -> str:
        """Helper function gets Cubename from a Path

        Args:
            path (Path): One of Path/to/cubename.npz and Path/to/experiment_cubename.npz

        Returns:
            [str]: cubename (has format tile_stuff.npz)
        """        
        components = path.name.split("_")
        regex = re.compile('\d{2}[A-Z]{3}')
        if bool(regex.match(components[0])):
            return path.name
        else:
            assert(bool(regex.match(components[1])))
            return "_".join(components[1:]) 

    def target_computation(self, minicube):
        """Compute the vegetation index (VI) target"""
        if self.target == "ndvi":
            targ = (minicube.s2_B08 - minicube.s2_B04) / (minicube.s2_BO8 + minicube.s2_B04 + 1e-6)
        
        if self.target == "kndvi": # TODO the the denominator is not optimal, needs to be improved accordingly to the original paper
            targ = np.tanh(((minicube.s2_B08 - minicube.s2_B04) / (minicube.s2_BO8 + minicube.s2_B04 + 1e-6))**2) / np.tanh(1) 

        # TODO add ndvi normalized, for each measurement, need to select the good month in ndviclim
        return targ



class EarthNet2023DataModule(pl.LightningDataModule):

    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters(copy.deepcopy(hparams))
        self.base_dir = Path(hparams.base_dir)
        
    @staticmethod
    def add_data_specific_args(parent_parser: Optional[Union[argparse.ArgumentParser,list]] = None):
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]
        
        parser = argparse.ArgumentParser(parents = parent_parser, add_help = False)

        parser.add_argument('--base_dir', type = str, default = "data/datasets/")
        parser.add_argument('--test_track', type = str, default = "iid")
        parser.add_argument('--target', type = str, default = "ndvi")

        parser.add_argument('--fp16', type = str2bool, default = False)

        parser.add_argument('--train_batch_size', type = int, default = 1)
        parser.add_argument('--val_batch_size', type = int, default = 1)
        parser.add_argument('--test_batch_size', type = int, default = 1)

        parser.add_argument('--val_split_seed', type = float, default = 42)

        parser.add_argument('--num_workers', type = int, default = multiprocessing.cpu_count())

        return parser
    
    def setup(self, stage: str = None):
        if stage == 'fit' or stage is None:
            self.earthnet_train = EarthNet2023Dataset(self.base_dir/"train", target=self.hparams.target, fp16=self.hparams.fp16)
            self.earthnet_val = EarthNet2023Dataset(self.base_dir/"val", target=self.hparams.target, fp16=self.hparams.fp16)

        if stage == 'test' or stage is None:
            self.earthnet_test = EarthNet2023Dataset(self.base_dir/"test", target=self.hparams.target, fp16=self.hparams.fp16)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.earthnet_train, batch_size=self.hparams.train_batch_size, num_workers=self.hparams.num_workers,pin_memory=True,drop_last=True, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.earthnet_val, batch_size=self.hparams.val_batch_size, num_workers=self.hparams.num_workers, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.earthnet_test, batch_size=self.hparams.test_batch_size, num_workers=self.hparams.num_workers, pin_memory=True)


     



