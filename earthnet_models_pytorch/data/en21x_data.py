import argparse
import copy
import multiprocessing
import re
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr
from earthnet_models_pytorch.utils import str2bool
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

warnings.simplefilter(action="ignore", category=FutureWarning)


class EarthNet2021XDataset(Dataset):

    def __init__(
        self,
        folder: Union[Path, str],
        fp16=False,
        s2_bands=["ndvi", "B02", "B03", "B04", "B8A"],
        eobs_vars=["fg", "hu", "pp", "qq", "rr", "tg", "tn", "tx"],
        eobs_agg=["mean", "min", "max"],
        static_vars=["nasa_dem", "alos_dem", "cop_dem", "esawc_lc", "geom_cls"],
        start_month_extreme=None,
        dl_cloudmask=False,
        allow_fastaccess=False,
    ):
        if not isinstance(folder, Path):
            folder = Path(folder)

        if (
            allow_fastaccess
            and (
                folder.stem
                in [
                    "train",
                    "test_chopped",
                    "iid_chopped",
                    "ood-t_chopped",
                    "ood-st_chopped",
                    "ood-s_chopped",
                    "iid",
                    "val_chopped",
                    "flux_chopped",
                ]
            )
            and (folder.parent / f"{folder.stem}_fastaccess").exists()
        ):
            folder = folder.parent / f"{folder.stem}_fastaccess"
            print("Fast Access Dataloading enabled")
            self.fast_access = True
            self.filepaths = sorted(list(folder.glob("**/*.npz")))
        else:
            self.fast_access = False
            self.filepaths = sorted(list(folder.glob("**/*.nc")))

        self.type = np.float16 if fp16 else np.float32

        self.s2_bands = s2_bands
        self.eobs_vars = eobs_vars
        self.eobs_agg = eobs_agg
        self.static_vars = static_vars
        self.start_month_extreme = start_month_extreme
        self.dl_cloudmask = dl_cloudmask

        self.eobs_mean = xr.DataArray(
            data=[
                8.90661030749754,
                2.732927619847993,
                77.54440854529798,
                1014.330962704611,
                126.47924227500346,
                1.7713217310829938,
                4.770701430461286,
                13.567999825718509,
            ],
            coords={
                "variable": [
                    "eobs_tg",
                    "eobs_fg",
                    "eobs_hu",
                    "eobs_pp",
                    "eobs_qq",
                    "eobs_rr",
                    "eobs_tn",
                    "eobs_tx",
                ]
            },
        )
        self.eobs_std = xr.DataArray(
            data=[
                9.75620252236597,
                1.4870108944469236,
                13.511387994026359,
                10.262645403460999,
                97.05522895011327,
                4.147967261223076,
                9.044987677752898,
                11.08198777356161,
            ],
            coords={
                "variable": [
                    "eobs_tg",
                    "eobs_fg",
                    "eobs_hu",
                    "eobs_pp",
                    "eobs_qq",
                    "eobs_rr",
                    "eobs_tn",
                    "eobs_tx",
                ]
            },
        )

        self.static_mean = xr.DataArray(
            data=[0.0, 0.0, 0.0, 0.0, 0.0],
            coords={
                "variable": ["nasa_dem", "alos_dem", "cop_dem", "esawc_lc", "geom_cls"]
            },
        )
        self.static_std = xr.DataArray(
            data=[500.0, 500.0, 500.0, 1.0, 1.0],
            coords={
                "variable": ["nasa_dem", "alos_dem", "cop_dem", "esawc_lc", "geom_cls"]
            },
        )

        print(f"dataset has {len(self)} samples")

    def __getitem__(self, idx: int) -> dict:

        filepath = self.filepaths[idx]

        if self.fast_access:

            npz = np.load(filepath)
            data = {
                "dynamic": [
                    torch.from_numpy(npz["sen2arr"].astype(self.type)),
                    torch.from_numpy(npz["eobsarr"].astype(self.type)),
                ],
                "dynamic_mask": [torch.from_numpy(npz["sen2mask"].astype(self.type))],
                "static": [torch.from_numpy(npz["staticarr"].astype(self.type))],
                "static_mask": [],
                "landcover": torch.from_numpy(npz["lc"].astype(self.type)),
                "filepath": str(
                    filepath.parent.parent.parent
                    / f"{filepath.parent.parent.stem[:-11]}/"
                    / filepath.parent.stem
                    / f"{filepath.stem}.nc"
                ),
                "cubename": filepath.stem,
            }
            return data

        minicube = xr.open_dataset(filepath)

        if self.start_month_extreme:
            start_idx = {"march": 10, "april": 15, "may": 20, "june": 25, "july": 30}[
                self.start_month_extreme
            ]
            minicube = minicube.isel(time=slice(5 * start_idx, 5 * (start_idx + 30)))

        nir = minicube.s2_B8A
        red = minicube.s2_B04

        ndvi = (nir - red) / (nir + red + 1e-8)

        minicube["s2_ndvi"] = ndvi

        sen2arr = (
            minicube[[f"s2_{b}" for b in self.s2_bands]]
            .to_array("band")
            .isel(time=slice(4, None, 5))
            .transpose("time", "band", "lat", "lon")
            .values
        )

        sen2arr[np.isnan(sen2arr)] = 0.0  # Fill NaNs!!

        if self.dl_cloudmask:
            sen2mask = (
                minicube.s2_dlmask.where(
                    minicube.s2_dlmask > 0,
                    4 * (~minicube.s2_SCL.isin([1, 2, 4, 5, 6, 7])),
                )
                .isel(time=slice(4, None, 5))
                .transpose("time", "lat", "lon")
                .values[:, None, ...]
            )
            sen2mask[np.isnan(sen2mask)] = 4.0
        else:
            sen2mask = (
                minicube[["s2_mask"]]
                .to_array("band")
                .isel(time=slice(4, None, 5))
                .transpose("time", "band", "lat", "lon")
                .values
            )
            sen2mask[np.isnan(sen2mask)] = 4.0

        eobs = (
            (
                minicube[[f"eobs_{v}" for v in self.eobs_vars]].to_array("variable")
                - self.eobs_mean
            )
            / self.eobs_std
        ).transpose("time", "variable")

        eobsarr = []
        if "mean" in self.eobs_agg:
            eobsarr.append(eobs.coarsen(time=5, coord_func="max").mean())
        if "min" in self.eobs_agg:
            eobsarr.append(eobs.coarsen(time=5, coord_func="max").min())
        if "max" in self.eobs_agg:
            eobsarr.append(eobs.coarsen(time=5, coord_func="max").max())
        if "std" in self.eobs_agg:
            eobsarr.append(eobs.coarsen(time=5, coord_func="max").std())

        eobsarr = np.concatenate(eobsarr, axis=1)

        eobsarr[np.isnan(eobsarr)] = 0.0  # MAYBE BAD IDEA......

        staticarr = (
            (
                (minicube[self.static_vars].to_array("variable") - self.static_mean)
                / self.static_std
            )
            .transpose("variable", "lat", "lon")
            .values
        )

        staticarr[np.isnan(staticarr)] = 0.0  # MAYBE BAD IDEA......

        lc = (
            minicube[["esawc_lc"]]
            .to_array("variable")
            .transpose("variable", "lat", "lon")
            .values
        )  # c h w

        data = {
            "dynamic": [
                torch.from_numpy(sen2arr.astype(self.type)),
                torch.from_numpy(eobsarr.astype(self.type)),
            ],
            "dynamic_mask": [torch.from_numpy(sen2mask.astype(self.type))],
            "static": [torch.from_numpy(staticarr.astype(self.type))],
            "static_mask": [],
            "landcover": torch.from_numpy(lc.astype(self.type)),
            "filepath": str(filepath),
            "cubename": filepath.stem,
        }

        return data

    def __len__(self) -> int:
        return len(self.filepaths)


class EarthNet2021XDataModule(pl.LightningDataModule):
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

        parser.add_argument("--fp16", type=str2bool, default=False)
        parser.add_argument("--dl_cloudmask", type=str2bool, default=False)
        parser.add_argument("--allow_fastaccess", type=str2bool, default=False)
        parser.add_argument("--shuffle_train", type=str2bool, default=False)
        
        parser.add_argument('--new_valset', type = str2bool, default = False)
        parser.add_argument("--val_pct", type=float, default=0.05)
        parser.add_argument("--val_split_seed", type=float, default=42)

        parser.add_argument("--train_batch_size", type=int, default=1)
        parser.add_argument("--val_batch_size", type=int, default=1)
        parser.add_argument("--test_batch_size", type=int, default=1)

        parser.add_argument(
            "--num_workers", type=int, default=multiprocessing.cpu_count()
        )

        return parser

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:

            if self.hparams.new_valset:
                self.earthnet_train = EarthNet2021XDataset(
                    self.base_dir / "train",
                    fp16=self.hparams.fp16,
                    dl_cloudmask=self.hparams.dl_cloudmask,
                    allow_fastaccess=self.hparams.allow_fastaccess,
                )
                self.earthnet_val = EarthNet2021XDataset(
                    self.base_dir / "val_chopped/",
                    fp16=self.hparams.fp16,
                    dl_cloudmask=self.hparams.dl_cloudmask,
                    allow_fastaccess=self.hparams.allow_fastaccess,
                )
            else:
                earthnet_corpus = EarthNet2021XDataset(
                    self.base_dir / "train",
                    fp16=self.hparams.fp16,
                    dl_cloudmask=self.hparams.dl_cloudmask,
                    allow_fastaccess=self.hparams.allow_fastaccess,
                )

                val_size = int(self.hparams.val_pct * len(earthnet_corpus))
                train_size = len(earthnet_corpus) - val_size

                try:  # PyTorch 1.5 safe....
                    self.earthnet_train, self.earthnet_val = random_split(
                        earthnet_corpus,
                        [train_size, val_size],
                        generator=torch.Generator().manual_seed(
                            int(self.hparams.val_split_seed)
                        ),
                    )
                except TypeError:
                    self.earthnet_train, self.earthnet_val = random_split(
                        earthnet_corpus, [train_size, val_size]
                    )

        if stage == "test" or stage is None:
            if self.hparams.test_track.startswith("extreme_"):
                start_month_extreme = self.hparams.test_track.split("_")[-1]
                self.earthnet_test = EarthNet2021XDataset(
                    self.base_dir / "extreme",
                    fp16=self.hparams.fp16,
                    start_month_extreme=start_month_extreme,
                    dl_cloudmask=self.hparams.dl_cloudmask,
                    allow_fastaccess=self.hparams.allow_fastaccess,
                )  # , fg_masked = self.hparams.fg_masked)
            else:
                self.earthnet_test = EarthNet2021XDataset(
                    self.base_dir / self.hparams.test_track,
                    fp16=self.hparams.fp16,
                    dl_cloudmask=self.hparams.dl_cloudmask,
                    allow_fastaccess=self.hparams.allow_fastaccess,
                )  # , fg_masked = self.hparams.fg_masked)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.earthnet_train,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=self.hparams.shuffle_train,
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
