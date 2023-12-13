from typing import Union, Optional

import argparse
import copy
import multiprocessing
import re
import sys
import pandas as pd
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr

from torch.utils.data import Dataset, DataLoader, random_split

from earthnet_models_pytorch.utils import str2bool

variables = {
    ### variables with dimensions x, y, Sampled time (irregular)
    # Sentinel 2 bands. Spatio-temporal data. Every 5 days.
    "s2_bands": ["B02", "B03", "B04", "B05", "B06", "B07", "B8A"],
    # "s2_avail": ["s2_avail"],
    "s2_scene_classification": ["SCL"],
    # S2 EarthNet cloudmask
    "cloud_mask": ["cloudmask_en"],
    ### variables with dimension time
    # Era5 reanalysis dataset. Era5 pev has a low precision
    "era5": [
        "t2m_mean",
        "pev_mean",
        "slhf_mean",
        "ssr_mean",
        "sp_mean",
        "sshf_mean",
        "e_mean",
        "tp_mean",
        "t2m_min",
        "pev_min",
        "slhf_min",
        "ssr_min",
        "sp_min",
        "sshf_min",
        "e_min",
        "tp_min",
        "t2m_max",
        "pev_max",
        "slhf_max",
        "ssr_max",
        "sp_max",
        "sshf_max",
        "e_max",
        "tp_max",
    ],
    ### variables with dimension event_time
    "event" : [
        "events",
        "event_labels",
    ],
    ### static variables
    ### variables with dimensions x_300, y_300, 
    # Land cover classes. Categorical variable.
    "landcover": ["lccs_class"],
    ### variables with dimensions x, y
    # Elevation model. Defined on 0 - 2000.
    "elevation": ["cop_dem"], 
}


class DeepExtremes2023Dataset(Dataset):
    def __init__(
        self, folder: Union[Path, str], target: str, variables=variables, fp16=False
    ):
        if not isinstance(folder, Path):
            folder = Path(folder)
        self.filepaths = sorted(list(folder.glob("*.zarr")))  # why sorted?
        self.type = np.float16 if fp16 else np.float32
        self.target = target
        self.variables = variables

    def __getitem__(self, idx: int) -> dict:

        filepath = self.filepaths[idx]
        # print(filepath)
        minicube = xr.open_dataset(filepath)

        print(minicube.cloud_mask.time)
        print(minicube.s2_bands.B02.time)
        if minicube.cloud_mask.time != minicube.s2_bands.B02.time:
            raise Exception(
                "The first available imagery of sentinel-2 is not 4 + [5]"
                + str(index_avail)
            )


        # Create the minicube
        # s2 is 5 days, and already rescaled [0, 1]
        print(minicube[self.variables["s2_bands"]].shape)
        s2_cube = (
            minicube[self.variables["s2_bands"]]
            .to_array()
            .values.transpose((3, 0, 1, 2))
            .astype(self.type)
        )  # (time, channels, w, h)
        print(s2_cube)

        # s2_mask: 
        # 0 - free_sky
        # 1 - cloud
        # 2 - cloud_shadows
        # 3 - snow
        # 4 - masked_other_reasons

        s2_mask = (
            minicube[self.variables["cloud_mask"]]
            .to_array()
            .isel(time=time)
            .values.transpose((1, 0, 2, 3))
            .astype(self.type)[:, :, :128, :128]
        )  # (time, 1, w, h)

        target = (
            self.target_computation(minicube)
            .isel(time=time)
            .values[:, None, ...]
            .astype(self.type)[:, :, :128, :128]
        )

        # weather is daily
        meteo_cube = minicube[self.variables["era5"]]

        # rescale temperature on the extreme values ever observed globally (Kelvin): -88, 58.
        meteo_cube["t2m"] = (meteo_cube["t2m"] - 185) / (331 - 185)
        meteo_cube["t2mmin"] = (meteo_cube["t2mmin"] - 185) / (331 - 185)
        meteo_cube["t2mmax"] = (meteo_cube["t2mmax"] - 185) / (331 - 185)

        # shouldn't I rescale all meteo variables?
        meteo_cube = (
            meteo_cube.to_array().values.transpose((1, 0)).astype(self.type)
        )  # t, c
        meteo_cube[np.isnan(meteo_cube)] = 0

        # TODO NaN values are replaced by the mean of each variable. To solve, currently RuntimeWarning: overflow encountered in reduce
        # col_mean = np.nanmean(meteo_cube, axis=0)
        # inds = np.where(np.isnan(meteo_cube))
        # meteo_cube[inds] = np.take(col_mean, inds[1])

        topography = (
            minicube[self.variables["elevation"]].to_array() / 2000
        )  # c h w, rescaling
        topography = topography.values.astype(self.type)[:, :128, :128]
        topography[np.isnan(topography)] = np.mean(
            topography
        )  # idk if we can have missing value in the topography

        landcover = (
            minicube[self.variables["landcover"]]
            .to_array()
            .values.astype(self.type)[:, :128, :128]
        )  # c h w

        # TODO transform landcover in categoritcal variables if used for training

        # NaN values handling
        s2_cube = np.where(np.isnan(s2_cube), np.zeros(1).astype(self.type), s2_cube)
        target = np.where(np.isnan(target), np.zeros(1).astype(self.type), target)
        s2_mask = np.where(np.isnan(s2_mask), np.ones(1).astype(self.type), s2_mask) # ?? s2_cube ? or s2_mask ?

        landcover = np.where(
            np.isnan(landcover), np.zeros(1).astype(self.type), landcover
        )

        # include scene classification in model? https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm-overview
        # Concatenation
        satellite_data = np.concatenate((target, s2_cube), axis=1)

        # Final minicube
        data = {
            "dynamic": [
                torch.from_numpy(satellite_data),
                torch.from_numpy(meteo_cube),
            ],
            "dynamic_mask": [torch.from_numpy(s2_mask)],
            "static": [torch.from_numpy(topography)],
            # "target": torch.from_numpy(target),
            # "s2_avail": torch.from_numpy(s2_avail),
            "landcover": torch.from_numpy(landcover),
            "filepath": str(filepath),
            "cubename": self.__name_getter(filepath),
        }
        # print(data["filepath"], data["dynamic"][0].shape)
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
        regex = re.compile("\d{2}[A-Z]{3}")
        if bool(regex.match(components[0])):
            return path.name
        else:
            assert bool(regex.match(components[1]))
            return "_".join(components[1:])

    def target_computation(self, minicube):
        """Compute the vegetation index (VI) target"""
        if self.target == "ndvi":
            targ = (minicube.B8A - minicube.B04) / (
                minicube.B8A + minicube.B04 + 1e-6
            )

        if (
            self.target == "kndvi"
        ):  # TODO the the denominator is not optimal, needs to be improved accordingly to the original paper
            targ = np.tanh(
                (
                    (minicube.B08 - minicube.B04)
                    / (minicube.BO8 + minicube.B04 + 1e-6)
                )
                ** 2
            ) / np.tanh(1)

        # if self.target == "anomalie_ndvi":
        #     targ = (minicube.B8A - minicube.B04) / (
        #         minicube.B8A + minicube.B04 + 1e-6
        #     )
        #     for i in range(1, 13):
        #         indices = targ.groupby("time.month").groups[i]
        #         index_month = minicube.ndviclim_mean.groupby("time_clim.month").groups[
        #             i
        #         ]
        #         targ[indices] = (
        #             targ[indices].values - minicube.ndviclim_mean[index_month].values
        #         )

        return targ


class DeepExtremes2023DataModule(pl.LightningDataModule):
    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters(copy.deepcopy(hparams))
        self.base_dir = Path(hparams.base_dir)

    @staticmethod
    def add_data_specific_args(
        parent_parser: Optional[Union[argparse.ArgumentParser, list]] = None
    ):
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]

        parser = argparse.ArgumentParser(parents=parent_parser, add_help=False)

        parser.add_argument("--base_dir", type=str, default="data/datasets/")
        parser.add_argument("--test_track", type=str, default="iid")
        parser.add_argument("--target", type=str, default="ndvi")

        parser.add_argument("--fp16", type=str2bool, default=False)

        parser.add_argument("--train_batch_size", type=int, default=1)
        parser.add_argument("--val_batch_size", type=int, default=1)
        parser.add_argument("--test_batch_size", type=int, default=1)

        parser.add_argument("--val_split_seed", type=int, default=42)

        parser.add_argument(
            "--num_workers", type=int, default=multiprocessing.cpu_count()
        )

        return parser

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            earthnet_full = DeepExtremes2023Dataset(
                self.base_dir, # / "train",
                target=self.hparams.target,
                fp16=self.hparams.fp16,
            )
            self.earthnet_train, self.earthnet_val = random_split(
                earthnet_full,
                [0.95, 0.05],
                generator=torch.Generator().manual_seed(self.hparams.val_split_seed),
            )

        if stage == "test" or stage is None:
            self.earthnet_test = DeepExtremes2023Dataset(
                self.base_dir / "test",
                target=self.hparams.target,
                fp16=self.hparams.fp16,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.earthnet_train,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.earthnet_val,
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.earthnet_test,
            batch_size=self.hparams.test_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
