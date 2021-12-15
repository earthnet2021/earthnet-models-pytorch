
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

class EarthNet2021XDataset(Dataset):

    def __init__(self, folder: Union[Path, str], fp16 = False, spatial_eobs = True, eobs_spread = False, soilgrids_all = False):
        if not isinstance(folder, Path):
            folder = Path(folder)

        self.filepaths = sorted(list(folder.glob("**/*.nc")))
        
        self.type = np.float16 if fp16 else np.float32

        self.spatial_eobs = spatial_eobs

        self.eobs_spread = eobs_spread

        self.soilgrids_all = soilgrids_all

        self.eobs_scaling_cube = xr.DataArray(data = [100, 100, 100, 20, 100, 100, 100, 100, 1000, 1000, 1000, 200, 1000, 1000, 1000, 200, 50, 50, 50, 10, 50, 50, 50, 10, 50, 50, 50, 10], coords = {"variable": ['fg_max', 'fg_mean', 'fg_min', 'fg_spread', 'hu_max', 'hu_mean', 'hu_min', 'hu_spread', 'qq_max',
    'qq_mean', 'qq_min', 'qq_spread', 'rr_max', 'rr_mean', 'rr_min', 'rr_spread', 'tg_max', 'tg_mean', 'tg_min', 'tg_spread', 'tn_max', 'tn_mean', 'tn_min', 'tn_spread', 'tx_max', 'tx_mean', 'tx_min', 'tx_spread']})

        self.eobs_vars = ['fg_max', 'fg_mean', 'fg_min', 'fg_spread', 'hu_max', 'hu_mean', 'hu_min', 'hu_spread', 'qq_max',
    'qq_mean', 'qq_min', 'qq_spread', 'rr_max', 'rr_mean', 'rr_min', 'rr_spread', 'tg_max', 'tg_mean', 'tg_min', 'tg_spread', 'tn_max', 'tn_mean', 'tn_min', 'tn_spread', 'tx_max', 'tx_mean', 'tx_min', 'tx_spread']

    def __getitem__(self, idx: int) -> dict:
        
        filepath = self.filepaths[idx]

        minicube = xr.open_dataset(filepath)

        kndvi = minicube.kndvi.values.transpose((2,0,1))[:, None, ...].astype(self.type) # t c h w

        kndvi /= np.tanh(1)

        kndvi[np.isnan(kndvi)] = 0

        kndvi[kndvi > 1] = 1
        kndvi[kndvi < 0] = 0

        eobs_cube = minicube[self.eobs_vars].to_array()

        eobs_cube  = eobs_cube / self.eobs_scaling_cube

        if not self.eobs_spread:
            eobs_cube = eobs_cube.sel(variable = [v for v in self.eobs_vars if "spread" not in v])

        if self.spatial_eobs:
            eobs = eobs_cube.values.transpose((1,0,2,3)).astype(self.type) # t c h w
        else:
            eobs = eobs_cube.values.transpose((1,0)).astype(self.type)

        eobs[np.isnan(eobs)] = 0  # MAYBE BAD IDEA......
        
        dem = minicube.dem.values[None,...].astype(self.type) # c h w
        
        dem /= 2000
        dem[np.isnan(dem)] = 0  # MAYBE BAD IDEA......

        if self.soilgrids_all:
            sg_cube = minicube[sorted([v for v in minicube.variables if v.startswith("sg")])].to_array()
        else:
            sg_cube = minicube[sorted([v for v in minicube.variables if v.startswith("sg")])]
            for var in ["bdod", "cec", "cfvo", "clay", "nitrogen", "phh2o", "ocd", "sand", "silt", "soc"]:
                sg_cube[f"sg_{var}_0-30cm_median"] = sg_cube[f"sg_{var}_0-5cm_Q0.5"] / 6 + sg_cube[f"sg_{var}_5-15cm_Q0.5"] / 3 + sg_cube[f"sg_{var}_15-30cm_Q0.5"] / 2
                sg_cube[f"sg_{var}_30-200cm_median"] = (3/17) * sg_cube[f"sg_{var}_30-60cm_Q0.5"] + (4/17)*sg_cube[f"sg_{var}_60-100cm_Q0.5"] + (10/17)*sg_cube[f"sg_{var}_100-200cm_Q0.5"]
            sg_cube = sg_cube[sorted([v for v in sg_cube.variables if v.endswith("median")])].to_array()


        sg = sg_cube.values.astype(self.type) # c h w

        sg /= 1000
        sg[np.isnan(sg)] = 0  # MAYBE BAD IDEA......

        highresstatic = np.concatenate([dem, sg], axis = 0)

        lc = minicube.lc.values[None, ...].astype(self.type) # c h w

        lc[np.isnan(lc)] = 0


        data = {
            "dynamic": [
                torch.from_numpy(kndvi),
                torch.from_numpy(eobs)
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


class EarthNet2021XpxDataset(Dataset):
    def __init__(self, folder: Union[Path, str], fp16 = False, eobs_spread = False, soilgrids_all = False):
        if not isinstance(folder, Path):
            folder = Path(folder)

        self.dataset_path = folder/"pixels_train.nc"

        self.dataset = xr.load_dataset(self.dataset_path)
        
        self.type = np.float16 if fp16 else np.float32

        self.eobs_spread = eobs_spread

        self.soilgrids_all = soilgrids_all

        self.eobs_scaling_cube = xr.DataArray(data = [100, 100, 100, 20, 100, 100, 100, 100, 1000, 1000, 1000, 200, 1000, 1000, 1000, 200, 50, 50, 50, 10, 50, 50, 50, 10, 50, 50, 50, 10], coords = {"variable": ['fg_max', 'fg_mean', 'fg_min', 'fg_spread', 'hu_max', 'hu_mean', 'hu_min', 'hu_spread', 'qq_max',
    'qq_mean', 'qq_min', 'qq_spread', 'rr_max', 'rr_mean', 'rr_min', 'rr_spread', 'tg_max', 'tg_mean', 'tg_min', 'tg_spread', 'tn_max', 'tn_mean', 'tn_min', 'tn_spread', 'tx_max', 'tx_mean', 'tx_min', 'tx_spread']})

        self.eobs_vars = ['fg_max', 'fg_mean', 'fg_min', 'fg_spread', 'hu_max', 'hu_mean', 'hu_min', 'hu_spread', 'qq_max',
    'qq_mean', 'qq_min', 'qq_spread', 'rr_max', 'rr_mean', 'rr_min', 'rr_spread', 'tg_max', 'tg_mean', 'tg_min', 'tg_spread', 'tn_max', 'tn_mean', 'tn_min', 'tn_spread', 'tx_max', 'tx_mean', 'tx_min', 'tx_spread']

        print("Initialized dataset")

    def __getitem__(self, idx: int) -> dict:

        pixel = self.dataset.isel(loc = idx)

        kndvi = pixel.kndvi.values[:,None].astype(self.type) # t c

        kndvi /= np.tanh(1)

        kndvi[np.isnan(kndvi)] = 0

        kndvi[kndvi > 1] = 1
        kndvi[kndvi < 0] = 0

        eobs_cube = pixel[self.eobs_vars].to_array()

        eobs_cube  = eobs_cube / self.eobs_scaling_cube

        if not self.eobs_spread:
            eobs_cube = eobs_cube.sel(variable = [v for v in self.eobs_vars if "spread" not in v])

        eobs = eobs_cube.values.T.astype(self.type) # t c

        eobs[np.isnan(eobs)] = 0  # MAYBE BAD IDEA......

        dem = pixel.dem.values[None,...].astype(self.type) # c

        dem /= 2000
        dem[np.isnan(dem)] = 0  # MAYBE BAD IDEA......

        if self.soilgrids_all:
            sg_cube = pixel[sorted([v for v in pixel.variables if v.startswith("sg")])].to_array()
        else:
            sg_cube = pixel[sorted([v for v in pixel.variables if v.startswith("sg")])]
            for var in ["bdod", "cec", "cfvo", "clay", "nitrogen", "phh2o", "ocd", "sand", "silt", "soc"]:
                sg_cube[f"sg_{var}_0-30cm_median"] = sg_cube[f"sg_{var}_0-5cm_Q0.5"] / 6 + sg_cube[f"sg_{var}_5-15cm_Q0.5"] / 3 + sg_cube[f"sg_{var}_15-30cm_Q0.5"] / 2
                sg_cube[f"sg_{var}_30-200cm_median"] = (3/17) * sg_cube[f"sg_{var}_30-60cm_Q0.5"] + (4/17)*sg_cube[f"sg_{var}_60-100cm_Q0.5"] + (10/17)*sg_cube[f"sg_{var}_100-200cm_Q0.5"]
            sg_cube = sg_cube[sorted([v for v in sg_cube.variables if v.endswith("median")])].to_array()


        sg = sg_cube.values.astype(self.type) # c

        sg /= 1000
        sg[np.isnan(sg)] = 0  # MAYBE BAD IDEA......

        highresstatic = np.concatenate([dem, sg], axis = 0)

        lc = pixel.lc.values[None, ...].astype(self.type) # c

        lc[np.isnan(lc)] = 0

        data = {
            "dynamic": [
                torch.from_numpy(kndvi),
                torch.from_numpy(eobs)
            ],
            "dynamic_mask": [],
            "static": [
                torch.from_numpy(highresstatic)
            ],
            "static_mask": [],
            "landcover": torch.from_numpy(lc),
            "filepath": "",
            "cubename": str(pixel["loc"].values)
        }

        return data

    def __len__(self) -> int:
        return len(self.dataset["loc"])


class EarthNet2021XDataModule(pl.LightningDataModule):

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
        parser.add_argument('--spatial_eobs', type = str2bool, default = True)
        parser.add_argument('--eobs_spread', type = str2bool, default = True)
        parser.add_argument('--soilgrids_all', type = str2bool, default = True)
        parser.add_argument('--val_pct', type = float, default = 0.05)
        parser.add_argument('--val_split_seed', type = float, default = 42)

        parser.add_argument('--train_batch_size', type = int, default = 1)
        parser.add_argument('--val_batch_size', type = int, default = 1)
        parser.add_argument('--test_batch_size', type = int, default = 1)

        parser.add_argument('--num_workers', type = int, default = multiprocessing.cpu_count())

        return parser
    
    def setup(self, stage: str = None):

        if stage == 'fit' or stage is None:
            earthnet_corpus = EarthNet2021XDataset(self.base_dir/"train", fp16 = self.hparams.fp16, spatial_eobs = self.hparams.spatial_eobs, eobs_spread = self.hparams.eobs_spread, soilgrids_all = self.hparams.soilgrids_all)

            val_size = int(self.hparams.val_pct * len(earthnet_corpus))
            train_size = len(earthnet_corpus) - val_size

            try: #PyTorch 1.5 safe....
                self.earthnet_train, self.earthnet_val = random_split(earthnet_corpus, [train_size, val_size], generator=torch.Generator().manual_seed(int(self.hparams.val_split_seed)))
            except TypeError:
                self.earthnet_train, self.earthnet_val = random_split(earthnet_corpus, [train_size, val_size])

        if stage == 'test' or stage is None:
            self.earthnet_test = EarthNet2021XDataset(self.base_dir/"test"/self.hparams.test_track, fp16 = self.hparams.fp16, spatial_eobs = self.hparams.spatial_eobs, eobs_spread = self.hparams.eobs_spread, soilgrids_all = self.hparams.soilgrids_all)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.earthnet_train, batch_size=self.hparams.train_batch_size, num_workers = self.hparams.num_workers,pin_memory=True,drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.earthnet_val, batch_size=self.hparams.val_batch_size, num_workers = self.hparams.num_workers, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.earthnet_test, batch_size=self.hparams.test_batch_size, num_workers = self.hparams.num_workers, pin_memory=True)



class EarthNet2021XpxDataModule(pl.LightningDataModule):

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
        parser.add_argument('--spatial_eobs', type = str2bool, default = True)
        parser.add_argument('--eobs_spread', type = str2bool, default = True)
        parser.add_argument('--soilgrids_all', type = str2bool, default = True)
        parser.add_argument('--val_pct', type = float, default = 0.05)
        parser.add_argument('--val_split_seed', type = float, default = 42)

        parser.add_argument('--train_batch_size', type = int, default = 1)
        parser.add_argument('--val_batch_size', type = int, default = 1)
        parser.add_argument('--test_batch_size', type = int, default = 1)

        parser.add_argument('--num_workers', type = int, default = multiprocessing.cpu_count())

        return parser
    
    def setup(self, stage: str = None):

        if stage == 'fit' or stage is None:
            earthnet_corpus = EarthNet2021XDataset(self.base_dir/"train", fp16 = self.hparams.fp16, spatial_eobs = self.hparams.spatial_eobs, eobs_spread = self.hparams.eobs_spread, soilgrids_all = self.hparams.soilgrids_all)

            val_size = int(self.hparams.val_pct * len(earthnet_corpus))
            train_size = len(earthnet_corpus) - val_size

            try: #PyTorch 1.5 safe....
                _, self.earthnet_val = random_split(earthnet_corpus, [train_size, val_size], generator=torch.Generator().manual_seed(int(self.hparams.val_split_seed)))
            except TypeError:
                _, self.earthnet_val = random_split(earthnet_corpus, [train_size, val_size])

            self.earthnet_train = EarthNet2021XpxDataset(self.base_dir, fp16 = self.hparams.fp16, eobs_spread = self.hparams.eobs_spread, soilgrids_all = self.hparams.soilgrids_all)

        if stage == 'test' or stage is None:
            self.earthnet_test = EarthNet2021XDataset(self.base_dir/"test"/self.hparams.test_track, fp16 = self.hparams.fp16, spatial_eobs = self.hparams.spatial_eobs, eobs_spread = self.hparams.eobs_spread, soilgrids_all = self.hparams.soilgrids_all)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.earthnet_train, batch_size=self.hparams.train_batch_size, num_workers = self.hparams.num_workers,pin_memory=True,drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.earthnet_val, batch_size=self.hparams.val_batch_size, num_workers = self.hparams.num_workers, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.earthnet_test, batch_size=self.hparams.test_batch_size, num_workers = self.hparams.num_workers, pin_memory=True)