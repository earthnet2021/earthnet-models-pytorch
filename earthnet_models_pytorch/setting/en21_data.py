
from typing import Union, Optional

import argparse
import copy
import multiprocessing
import re

from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch

from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

from earthnet_models_pytorch.utils import str2bool

class EarthNet2021Dataset(Dataset):

    def __init__(self, folder: Union[Path, str], noisy_masked_pixels = False, use_meso_static_as_dynamic = False, fp16 = False, landcover_folder: Optional[Union[Path, str]] = None, remove_first_n_steps = 0):
        if not isinstance(folder, Path):
            folder = Path(folder)
        assert (not {"target","context"}.issubset(set([d.name for d in folder.glob("*") if d.is_dir()])))

        self.filepaths = sorted(list(folder.glob("**/*.npz")))

        if landcover_folder is not None:
            if not isinstance(landcover_folder, Path):
                landcover_folder = Path(landcover_folder)
            self.landcover_folder = landcover_folder
        else:
            self.landcover_folder = None

        self.noisy_masked_pixels = noisy_masked_pixels
        self.use_meso_static_as_dynamic = use_meso_static_as_dynamic
        self.type = np.float16 if fp16 else np.float32
        self.t_0 = remove_first_n_steps

    def __getitem__(self, idx: int) -> dict:
        
        filepath = self.filepaths[idx]

        npz = np.load(filepath)

        highresdynamic = np.transpose(npz["highresdynamic"],(3,2,0,1)).astype(self.type)[self.t_0:,...]
        highresstatic = np.transpose(npz["highresstatic"],(2,0,1)).astype(self.type)
        mesodynamic = np.transpose(npz["mesodynamic"],(3,2,0,1)).astype(self.type)[self.t_0*5:,...]
        mesostatic = np.transpose(npz["mesostatic"],(2,0,1)).astype(self.type)

        masks = ((1 - highresdynamic[:,-1,:,:])[:,np.newaxis,:,:]).repeat(4,1)

        images = highresdynamic[:,:4,:,:]
        
        images[np.isnan(images)] = 0
        images[images > 1] = 1
        images[images < 0] = 0
        mesodynamic[np.isnan(mesodynamic)] = 0
        highresstatic[np.isnan(highresstatic)] = 0
        mesostatic[np.isnan(mesostatic)] = 0

        if self.noisy_masked_pixels:            
            images = np.transpose(images,(1,0,2,3))
            all_pixels = images[np.transpose(masks, (1,0,2,3)) == 1].reshape(4,-1)
            all_pixels = np.stack(int(images.size/all_pixels.size+1)*[all_pixels],axis = 1)
            all_pixels = all_pixels.reshape(4,-1)
            all_pixels = all_pixels.transpose(1,0)
            np.random.shuffle(all_pixels)
            all_pixels = all_pixels.transpose(1,0)
            all_pixels = all_pixels[:,:images.size//4].reshape(*images.shape)
            images = np.where(np.transpose(masks, (1,0,2,3)) == 0, all_pixels, images)
            images = np.transpose(images,(1,0,2,3))

        if self.use_meso_static_as_dynamic:
            mesodynamic = np.concatenate([mesodynamic, mesostatic[np.newaxis, :, :, :].repeat(mesodynamic.shape[0], 0)], axis = 1)

        data = {
            "dynamic": [
                torch.from_numpy(images),
                torch.from_numpy(mesodynamic)
            ],
            "dynamic_mask": [
                torch.from_numpy(masks)
            ],
            "static": [
                torch.from_numpy(highresstatic),
                torch.from_numpy(mesostatic)
            ] if not self.use_meso_static_as_dynamic else [
                torch.from_numpy(highresstatic)
            ],
            "static_mask": [],
            "filepath": str(filepath),
            "cubename": self.__name_getter(filepath)
        }

        if self.landcover_folder is not None:
            tile = filepath.parts[-2]
            filename = self.__name_getter(filepath)
            landcover_path = self.landcover_folder/tile/filename
            lc = np.load(landcover_path)["landcover"]
            data["landcover"] = torch.from_numpy(lc)

        npz.close()

        return data

    def __len__(self) -> int:
        return len(self.filepaths)

    def __name_getter(self, path: Path) -> str:
        """Helper function gets Cubename from a Path

        Args:
            path (Path): One of Path/to/cubename.npz and Path/to/experiment_cubename.npz

        Returns:
            [str]: cubename (has format tile_startyear_startmonth_startday_endyear_endmonth_endday_hrxmin_hrxmax_hrymin_hrymax_mesoxmin_mesoxmax_mesoymin_mesoymax.npz)
        """        
        components = path.name.split("_")
        regex = re.compile('\d{2}[A-Z]{3}')
        if bool(regex.match(components[0])):
            return path.name
        else:
            assert(bool(regex.match(components[1])))
            return "_".join(components[1:]) 

class EarthNet2021DataModule(pl.LightningDataModule):

    __TRACKS__ = {
        "iid": ("iid_test_split/context/","iid_test"),
        "ood": ("ood_test_split/context/","ood_test"),
        "ex": ("extreme_test_split/context/","extreme_test"),
        "sea": ("seasonal_test_split/context/", "seasonal_test"),
        "full_sea": ("seasonal_test_filtered", "seasonal_test")
    }

    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        if hasattr(self, "save_hyperparameters"):
            self.save_hyperparameters(copy.deepcopy(hparams))
        else:
            self.hparams = copy.deepcopy(hparams)
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

        parser.add_argument('--noisy_masked_pixels', type = str2bool, default = True)
        parser.add_argument('--use_meso_static_as_dynamic', type = str2bool, default = True)
        parser.add_argument('--fp16', type = str2bool, default = False)
        parser.add_argument('--landcover_folder', type = str, default = None)
        parser.add_argument('--val_pct', type = float, default = 0.05)
        parser.add_argument('--val_split_seed', type = float, default = 42)

        parser.add_argument('--train_batch_size', type = int, default = 1)
        parser.add_argument('--val_batch_size', type = int, default = 1)
        parser.add_argument('--test_batch_size', type = int, default = 1)

        parser.add_argument('--num_workers', type = int, default = multiprocessing.cpu_count())

        parser.add_argument('--full_sea_remove_first_n_steps', type = int, default = 0)

        return parser
    
    def setup(self, stage: str = None):

        if stage == 'fit' or stage is None:
            earthnet_corpus = EarthNet2021Dataset(self.base_dir/"train", noisy_masked_pixels = self.hparams.noisy_masked_pixels, use_meso_static_as_dynamic = self.hparams.use_meso_static_as_dynamic, fp16 = self.hparams.fp16, landcover_folder=Path(self.hparams.landcover_folder)/"train" if self.hparams.landcover_folder is not None else None)

            val_size = int(self.hparams.val_pct * len(earthnet_corpus))
            train_size = len(earthnet_corpus) - val_size

            try: #PyTorch 1.5 safe....
                self.earthnet_train, self.earthnet_val = random_split(earthnet_corpus, [train_size, val_size], generator=torch.Generator().manual_seed(int(self.hparams.val_split_seed)))
            except TypeError:
                self.earthnet_train, self.earthnet_val = random_split(earthnet_corpus, [train_size, val_size])

        if stage == 'test' or stage is None:
            self.earthnet_test = EarthNet2021Dataset(self.base_dir/self.__TRACKS__[self.hparams.test_track][0], noisy_masked_pixels = self.hparams.noisy_masked_pixels, use_meso_static_as_dynamic = self.hparams.use_meso_static_as_dynamic, fp16 = self.hparams.fp16, landcover_folder=Path(self.hparams.landcover_folder)/self.__TRACKS__[self.hparams.test_track][1] if self.hparams.landcover_folder is not None else None, remove_first_n_steps = 0 if self.hparams.test_track != "full_sea" else self.hparams.full_sea_remove_first_n_steps)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.earthnet_train, batch_size=self.hparams.train_batch_size, num_workers = self.hparams.num_workers,pin_memory=True,drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.earthnet_val, batch_size=self.hparams.val_batch_size, num_workers = self.hparams.num_workers, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.earthnet_test, batch_size=self.hparams.test_batch_size, num_workers = self.hparams.num_workers, pin_memory=True)