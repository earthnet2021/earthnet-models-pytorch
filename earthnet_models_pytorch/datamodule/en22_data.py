
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


class EarthNet2022Dataset(Dataset):

    def __init__(self, folder: Union[Path, str], fp16 = False):
        if not isinstance(folder, Path):
            folder = Path(folder)

        self.filepaths = sorted(list(folder.glob("**/*.nc")))
        
        self.type = np.float16 if fp16 else np.float32

        self.bands = ['ndvi', 'blue', 'green', 'red', 'nir']

        '''
        With soil moisture
        self.meteo_vars = ['pev_max', 'pev_mean', 'pev_min', 'sm_rootzone_max', 'sm_rootzone_mean', 'sm_rootzone_min', 'sm_surface_max', 'sm_surface_mean', 'sm_surface_min', 'sp_max', 'sp_mean', 'sp_min', 'ssr_max', 'ssr_mean', 'ssr_min', 'surface_pressure_max', 'surface_pressure_mean', 'surface_pressure_min', 't2m_max', 't2m_mean', 't2m_min', 'tp_max', 'tp_mean', 'tp_min']

        self.meteo_scaling_cube = xr.DataArray(data = [1e-4, 1e-4, 1e-4, 1, 1, 1, 1, 1, 1, 1000, 1000, 1000, 50, 50, 50, 1000, 1000, 1000, 400, 400, 400, 500, 500, 500], coords = {"variable": self.meteo_vars})
        
        mean_meteo = torch.tensor([-1.3039e+00, -1.8422e+00, -2.3154e+00,  2.4223e-01,  2.3511e-01,
         2.2876e-01,  2.2300e-01,  1.9395e-01,  1.6921e-01,  8.9678e-02,
         8.9426e-02,  8.9150e-02,  3.8471e-01,  3.2978e-01,  2.4605e-01,
         7.0871e-01,  7.0676e-01,  7.0464e-01,  7.3636e-01,  7.3166e-01,
         7.2685e-01,  2.0101e-02,  6.4336e-03,  9.6704e-04])

        

        std_meteo = torch.tensor([0.8378, 0.8577, 0.9764, 
         0.0883, 0.0863, 0.0846, 0.1034, 0.1033, 0.1037,
         0.0080, 0.0079, 0.0078, 0.0842, 0.0807, 0.0988, 
         0.0688, 0.0681, 0.0674,
         0.0146, 0.0144, 0.0153, 0.0294, 0.0105, 0.0029])
        '''
        
        self.meteo_vars = ['pev_max', 'pev_mean', 'pev_min', 'sp_max', 'sp_mean', 'sp_min', 'ssr_max', 'ssr_mean', 'ssr_min', 't2m_max', 't2m_mean', 't2m_min', 'tp_max', 'tp_mean', 'tp_min']
        self.meteo_scaling_cube = xr.DataArray(data = [1e-4, 1e-4, 1e-4, 1000, 1000, 1000, 50, 50, 50, 400, 400, 400, 500, 500, 500], coords = {"variable": self.meteo_vars})

        mean_meteo = torch.tensor([-1.3039e+00, -1.8422e+00, -2.3154e+00,  
         8.9678e-02,
         8.9426e-02,  8.9150e-02,  3.8471e-01,  3.2978e-01,  2.4605e-01,
         7.3636e-01,  7.3166e-01,
         7.2685e-01,  2.0101e-02,  6.4336e-03,  9.6704e-04])

        std_meteo = torch.tensor([0.8378, 0.8577, 0.9764, 
        0.0080, 0.0079, 0.0078, 0.0842, 0.0807, 0.0988,
        0.0146, 0.0144, 0.0153, 0.0294, 0.0105, 0.0029])

        self.transform_meteo = transforms.Compose([torch.from_numpy,
                                    transforms.Lambda(lambda x: (x - mean_meteo) / std_meteo)])

    def __getitem__(self, idx: int) -> dict:
        
        filepath = self.filepaths[idx]
        minicube = xr.open_dataset(filepath)

        # minicube["kndvi"] = np.tanh(((minicube.nir - minicube.red)/(minicube.nir+minicube.red+1e-6))**2) / np.tanh(1)
        minicube["ndvi"] = (minicube.nir - minicube.red)/(minicube.nir+minicube.red+1e-6)

        hr_cube = minicube[self.bands].to_array()
        hr = hr_cube.values.transpose((3,0,1,2)).astype(self.type) # t c h w

        hr[np.isnan(hr)] = 0
        hr[hr > 1] = 1
        hr[hr < 0] = 0
        
        meteo_cube = minicube[self.meteo_vars].to_array()
        meteo_cube  = meteo_cube / self.meteo_scaling_cube
        meteo = meteo_cube.values.transpose((1,0)).astype(self.type)

        meteo[np.isnan(meteo)] =  0

        highresstatic = minicube.dem.values[None,...].astype(self.type) # c h w       
        highresstatic /= 2000
        highresstatic[np.isnan(highresstatic)] = 0

        lc = minicube.lc.values[None, ...].astype(self.type) # c h w
        lc[np.isnan(lc)] = 0
    
        
        data = {
            "dynamic": [
                torch.from_numpy(hr),
                self.transform_meteo(meteo)
            ],
            "dynamic_mask": [],
            "static": [
                torch.from_numpy(highresstatic)
            ],
            "static_mask": [],
            "landcover": torch.from_numpy(lc),
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


class EarthNet2022DataModule(pl.LightningDataModule):

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

        parser.add_argument('--fp16', type = str2bool, default = False)

        parser.add_argument('--train_batch_size', type = int, default = 1)
        parser.add_argument('--val_batch_size', type = int, default = 1)
        parser.add_argument('--test_batch_size', type = int, default = 1)

        parser.add_argument('--val_pct', type = float, default = 0.05)
        parser.add_argument('--val_split_seed', type = float, default = 42)

        parser.add_argument('--num_workers', type = int, default = multiprocessing.cpu_count())

        return parser
    
    def setup(self, stage: str = None):
        if stage == 'fit' or stage is None:
            self.earthnet_train = EarthNet2022Dataset(self.base_dir/"train", fp16 = self.hparams.fp16)

            self.earthnet_val = EarthNet2022Dataset(self.base_dir/"val", fp16 = self.hparams.fp16)

        if stage == 'test' or stage is None:
            self.earthnet_test = EarthNet2022Dataset(self.base_dir/"test", fp16 = self.hparams.fp16)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.earthnet_train, batch_size=self.hparams.train_batch_size, num_workers = self.hparams.num_workers,pin_memory=True,drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.earthnet_val, batch_size=self.hparams.val_batch_size, num_workers = self.hparams.num_workers, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.earthnet_test, batch_size=self.hparams.test_batch_size, num_workers = self.hparams.num_workers, pin_memory=True)


     



