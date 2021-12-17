
from typing import Union, Optional

import argparse
import copy
import multiprocessing
import re

from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr

from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

from earthnet_models_pytorch.utils import str2bool

class EarthNet2022Dataset(Dataset):

    def __init__(self, folder: Union[Path, str], fp16 = False):
        if not isinstance(folder, Path):
            folder = Path(folder)

        self.filepaths = sorted(list(folder.glob("**/*.nc")))
        
        self.type = np.float16 if fp16 else np.float32

        self.bands = ['kndvi', 'blue', 'green', 'red', 'nir']

        self.meteo_vars = ['e_max', 'e_mean', 'e_min', 'heat_flux_latent_max', 'heat_flux_latent_mean', 'heat_flux_latent_min', 'heat_flux_sensible_max', 'heat_flux_sensible_mean', 'heat_flux_sensible_min', 'pev_max', 'pev_mean', 'pev_min', 'sm_rootzone_max', 'sm_rootzone_mean', 'sm_rootzone_min', 'sm_surface_max', 'sm_surface_mean', 'sm_surface_min', 'sp_max', 'sp_mean', 'sp_min', 'ssr_max', 'ssr_mean', 'ssr_min', 'surface_pressure_max', 'surface_pressure_mean', 'surface_pressure_min', 't2m_max', 't2m_mean', 't2m_min', 'tp_max', 'tp_mean', 'tp_min']

        self.meteo_scaling_cube = xr.DataArray(data = [10, 10, 10, 3000, 3000, 3000, 3000, 3000, 3000, 1e-4, 1e-4, 1e-4, 1, 1, 1, 1, 1, 1, 1000, 1000, 1000, 50, 50, 50, 1000, 1000, 1000, 400, 400, 400, 500, 500, 500], coords = {"variable": self.meteo_vars})


    def __getitem__(self, idx: int) -> dict:
        
        filepath = self.filepaths[idx]

        minicube = xr.open_dataset(filepath)

        minicube["kndvi"] = np.tanh(((minicube.nir - minicube.red)/(minicube.nir+minicube.red+1e-6))**2) / np.tanh(1)

        hr_cube = minicube[self.bands].to_array()
        hr = hr_cube.values.transpose((3,0,1,2)).astype(self.type) # t c h w

        hr[np.isnan(hr)] = 0

        hr[hr > 1] = 1
        hr[hr < 0] = 0

        meteo_cube = minicube[self.meteo_vars].to_array()

        meteo_cube  = meteo_cube / self.meteo_scaling_cube

        meteo = meteo_cube.values.transpose((1,0)).astype(self.type)

        meteo[np.isnan(meteo)] = 0  # MAYBE BAD IDEA......
        
        dem = minicube.dem.values[None,...].astype(self.type) # c h w
        
        dem /= 2000
        dem[np.isnan(dem)] = 0  # MAYBE BAD IDEA......


        highresstatic = dem #np.concatenate([dem, sg], axis = 0)

        lc = minicube.lc.values[None, ...].astype(self.type) # c h w

        lc[np.isnan(lc)] = 0


        data = {
            "dynamic": [
                torch.from_numpy(hr),
                torch.from_numpy(meteo)
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

