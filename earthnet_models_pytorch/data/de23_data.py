from typing import Union, Optional

import argparse
import copy
import multiprocessing
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr
import datetime

from torch.utils.data import Dataset, DataLoader, random_split

from earthnet_models_pytorch.utils import str2bool

import json


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
    "event": [
        "events",
        "event_labels",
    ],
    ### static variables
    ### variables with dimensions x_300, y_300,
    # Land cover classes. Categorical variable.
    # use Scene classification instead
    "landcover": ["lccs_class"],
    ### variables with dimensions x, y
    # Elevation model. Defined on 0 - 2000.
    "elevation": ["cop_dem"],
}

# with open("./scripts/preprocessing/statistics_de23.json", "r") as f:
#         statistic = json.load(f)
statistic = {
    "t2m_mean": {
        "mean": 285.14280428589245,
        "min": 220.7626190185547,
        "max": 318.43597412109375,
    },
    "pev_mean": {
        "mean": -0.003933192138161199,
        "min": -0.049169037491083145,
        "max": 0.0005616411217488348,
    },
    "slhf_mean": {"mean": -2162931.7410918274, "min": -15086400.0, "max": 2074217.375},
    "ssr_mean": {"mean": 6422307.595667305, "min": -0.0625, "max": 22510174.0},
    "sp_mean": {"mean": 91747.36897021797, "min": 49799.546875, "max": 105716.3984375},
    "sshf_mean": {"mean": -1334688.0814032578, "min": -12454722.0, "max": 9606654.0},
    "e_mean": {
        "mean": -0.0008624358549881119,
        "min": -0.006032629404217005,
        "max": 0.0007265469757840037,
    },
    "tp_mean": {"mean": 0.0015462739210485358, "min": 0.0, "max": 0.09146501123905182},
    "t2m_min": {
        "mean": 278.7418223321622,
        "min": 214.3935089111328,
        "max": 310.7576904296875,
    },
    "pev_min": {
        "mean": -0.010466829879496528,
        "min": -0.1887352466583252,
        "max": 1.1321157217025757e-05,
    },
    "slhf_min": {"mean": -5363088.528838599, "min": -45605476.0, "max": 163808.0},
    "ssr_min": {"mean": 147228.9512738576, "min": -1.0, "max": 3258205.0},
    "sp_min": {"mean": 91157.6636795228, "min": 49396.6875, "max": 105117.6875},
    "sshf_min": {"mean": -4187134.392026041, "min": -26917700.0, "max": 742063.0},
    "e_min": {
        "mean": -0.002136333048618079,
        "min": -0.018236353993415833,
        "max": 5.7774828746914864e-05,
    },
    "tp_min": {"mean": 3.998941094013752e-06, "min": 0.0, "max": 0.006604030728340149},
    "t2m_max": {
        "mean": 292.271053668584,
        "min": 224.75808715820312,
        "max": 325.5970153808594,
    },
    "pev_max": {
        "mean": -9.391972474487043e-05,
        "min": -0.004983663558959961,
        "max": 0.02069767564535141,
    },
    "slhf_max": {"mean": -1998.1564242578777, "min": -1504041.5, "max": 13011428.0},
    "ssr_max": {"mean": 14938754.622621803, "min": -0.0625, "max": 31967088.0},
    "sp_max": {"mean": 92268.74949649484, "min": 50269.9453125, "max": 106309.046875},
    "sshf_max": {"mean": 787876.1880326597, "min": -2240390.5, "max": 42488880.0},
    "e_max": {
        "mean": -1.9515435042241873e-06,
        "min": -0.0006014241371303797,
        "max": 0.0046675666235387325,
    },
    "tp_max": {"mean": 0.008162581792683693, "min": 0.0, "max": 0.397117555141449},
}


class DeepExtremes2023Dataset(Dataset):
    def __init__(
        self,
        folder: Union[Path, str],
        metadata_files,
        target: str,
        variables=variables,
        fp16=False,
        is_lc_mask=True,
    ):
        if not isinstance(folder, Path):
            folder = Path(folder)
        self.metadata = sorted(
            [
                (
                    Path(folder, metadata_files["path"][idx][1:]),
                    metadata_files["start_date"][idx],
                    metadata_files["end_date"][idx],
                )
                for idx in metadata_files.index
            ]
        )  # why sorted?
        self.type = np.float16 if fp16 else np.float32
        self.target = target
        self.variables = variables
        self.is_lc_mask = is_lc_mask

    def __getitem__(self, idx: int) -> dict:
        filepath, start_date, end_date = self.metadata[idx]
        minicube = xr.open_dataset(filepath, engine="zarr").sel(
            time=slice(start_date, end_date),
            event_time=slice(start_date, min(end_date, pd.Timestamp(year=2021, month=12, day=31))),
        )

        if (
            minicube[self.variables["cloud_mask"]].time
            != minicube[self.variables["s2_bands"]].B02.time
        ).all():
            raise Exception(
                "The first available imagery of sentinel-2 is not 4 + [5]"
                + str(index_avail)
            )

        # Create the minicube
        # s2 is 5 days, and already rescaled [0, 1]
        s2_cube = (
            minicube[self.variables["s2_bands"]]
            .to_array()
            .values.transpose((1, 0, 2, 3))
            .astype(self.type)
        )  # (time, channels, w, h)

        # s2_mask:
        # 0 - free_sky
        # 1 - cloud
        # 2 - cloud_shadows
        # 3 - snow
        # 4 - masked_other_reasons

        s2_mask = (
            minicube[self.variables["cloud_mask"]]
            .to_array()
            .values.transpose((1, 0, 2, 3))
            .astype(self.type)
        )  # (time, 1, w, h)
        # mask = 1 where there is a cloud, 0 where data
        s2_mask = s2_mask != 0

        target = (
            self.target_computation(minicube).values[:, None, ...].astype(self.type)
        )

        # rescale all meteo variables?
        for variable in self.variables["era5"]:
            minicube[variable] = (minicube[variable] - statistic[variable]["min"]) / (
                statistic[variable]["max"] - statistic[variable]["min"]
            )

        # Era5land and Era5 dataset. Weather is 5-daily
        meteo_cube = (
            minicube[self.variables["era5"]]
            .to_array()
            .values.transpose((1, 0))
            .astype(self.type)
        )

        topography = (
            minicube[self.variables["elevation"]].to_array().values.astype(self.type)
            / 2000
        )  # c h w, rescaling

        # SCL is scene classification. i.e., it has a time dimension.
        s2_scene_classification = (
            minicube[self.variables["s2_scene_classification"]]
            .to_array()
            .values.transpose((1, 0, 2, 3))
            .astype(self.type)
        )  # c h w

        # NaN values handling
        s2_cube = np.where(np.isnan(s2_cube), np.zeros(1).astype(self.type), s2_cube)
        target = np.where(np.isnan(target), np.zeros(1).astype(self.type), target)
        s2_mask = np.where(
            np.isnan(s2_mask), np.ones(1).astype(self.type), s2_mask
        )  # ?? s2_cube ? or s2_mask ?
        topography = np.where(
            np.isnan(topography), np.zeros(1).astype(self.type), topography
        )  # ?? s2_cube ? or s2_mask
        meteo_cube = np.where(
            np.isnan(meteo_cube), np.zeros(1).astype(self.type), meteo_cube
        )  # ?? s2_cube ? or s2_mask

        s2_scene_classification = np.where(
            np.isnan(s2_scene_classification),
            np.zeros(1).astype(self.type),
            s2_scene_classification,
        )

        # lc mask = 1 where data
        if self.is_lc_mask:
            lc_mask = (s2_scene_classification == 4).astype(  # 4 is vegetation pixel
                self.type
            )
        else:
            # no lc mask
            lc_mask = 1

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
            "landcover": torch.from_numpy(s2_scene_classification),
            "landcover_mask": torch.from_numpy(lc_mask).bool(),
            "filepath": str(filepath),
            "cubename": self.__name_getter(filepath, start_date),
        }
        return data

    def __len__(self) -> int:
        return len(self.metadata)

    def __name_getter(self, path: Path, start_date) -> str:
        """Helper function gets Cubename from a Path

        Args:
            path (Path): One of Path/to/cubename.npz and Path/to/experiment_cubename.npz

        Returns:
            [str]: cubename (has format tile_stuff.npz)
        """
        components = path.name.split("/")

        components1 = components[-1].split("_")
        name = "_".join(components1[0:3])
        output = name + "_" + str(start_date)[0:10]
        return output

    def target_computation(self, minicube) -> str:
        """Compute the vegetation index (VI) target"""
        if self.target == "ndvi":
            targ = (minicube.B8A - minicube.B04) / (minicube.B8A + minicube.B04 + 1e-6)

        if (
            self.target == "kndvi"
        ):  # TODO the denominator is not optimal, needs to be improved accordingly to the original paper
            targ = np.tanh(
                ((minicube.B08 - minicube.B04) / (minicube.BO8 + minicube.B04 + 1e-6))
                ** 2
            ) / np.tanh(1)

        if self.target == "anomalie_ndvi":
            targ = (
                (minicube.B8A - minicube.B04) / (minicube.B8A + minicube.B04 + 1e-6)
            ) - minicube.msc

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
        parser.add_argument("--fold_path", type=str, default="mc_earthnet.csv")
        parser.add_argument("--test_fold", type=int, default=10)
        parser.add_argument("--val_fold", type=int, default=9)

        parser.add_argument("--continent_split", type=bool, default=False)
        parser.add_argument("--test_track", type=str, default="iid")
        parser.add_argument("--target", type=str, default="ndvi")

        parser.add_argument("--context_length", type=int, default=72)
        parser.add_argument("--target_length", type=int, default=18)

        parser.add_argument("--is_lc_mask", type=bool, default=True)

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
        if self.hparams.continent_split:
            (
                train_subset,
                val_subset,
                spatial_test_subset,
                temporal_test_subset,
                continent_test_subset,
            ) = self.dataset_split_continent()
        else:
            (
                train_subset,
                val_subset,
                spatial_test_subset,
                temporal_test_subset,
            ) = self.dataset_split()

        if stage == "fit" or stage is None:
            self.earthnet_train = DeepExtremes2023Dataset(
                self.base_dir,
                train_subset,
                target=self.hparams.target,
                fp16=self.hparams.fp16,
                is_lc_mask=self.hparams.is_lc_mask,
            )

            self.earthnet_val = DeepExtremes2023Dataset(
                self.base_dir,
                val_subset,
                target=self.hparams.target,
                fp16=self.hparams.fp16,
                is_lc_mask=self.hparams.is_lc_mask,
            )

        if stage == "test" or stage is None:
            if self.hparams.test_track == "iid":
                self.earthnet_test = DeepExtremes2023Dataset(
                    self.base_dir,
                    spatial_test_subset,
                    target=self.hparams.target,
                    fp16=self.hparams.fp16,
                    is_lc_mask=self.hparams.lc_mask,
                )
            elif self.hparams.test_track == "temporal":
                self.earthnet_test = DeepExtremes2023Dataset(
                    self.base_dir,
                    temporal_test_subset,
                    target=self.hparams.target,
                    fp16=self.hparams.fp16,
                    is_lc_mask=self.hparams.lc_mask,
                )

            elif self.hparams.continent_split & (
                self.hparams.test_track == "continent"
            ):
                self.earthnet_test = DeepExtremes2023Dataset(
                    self.base_dir,
                    continent_test_subset,
                    target=self.hparams.target,
                    fp16=self.hparams.fp16,
                    is_lc_mask=self.hparams.lc_mask,
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

    def dataset_split(self):
        test_fold = self.hparams.test_fold
        val_fold = self.hparams.val_fold

        # load csv
        df = pd.read_csv(self.hparams.fold_path, delimiter=",")[
            ["path", "group", "check", "start_date"]
        ]

        # 3 training minicubes between 2017 - 2020 (randomly starting in the first year), followed by 3 test minicubes in 2021
        df["start_date0"] = pd.to_datetime(
            df["start_date"], format="%Y-%m-%dT%H:%M:%S.%f"
        )
        df = df.drop("start_date", axis=1)
        df["start_date2"] = df["start_date0"] + datetime.timedelta(days=(self.hparams.target_length + self.hparams.context_length)*5)
        df["start_date3"] = df["start_date2"] + datetime.timedelta(days=(self.hparams.target_length + self.hparams.context_length)*5)
        df["start_test1"] = df["start_date3"] + datetime.timedelta(days=(self.hparams.target_length + self.hparams.context_length)*5)
        df["start_test2"] = df["start_test1"] + datetime.timedelta(days=self.hparams.target_length * 5)
        df["start_test3"] = df["start_test2"] + datetime.timedelta(days=self.hparams.target_length * 5)
        df = df.melt(["path", "group", "check"], value_name="start_date")
        df["end_date"] = df["start_date"] + datetime.timedelta(days=(self.hparams.target_length + self.hparams.context_length) * 5 - 1 )

        # temporal test set 2021
        temporal_test_subset = df.loc[
            (df["variable"].str.startswith("start_test")) & (df["check"] == 0),
            ["path", "start_date", "end_date"],
        ].sort_values(by=["path","start_date"])
        # print(temporal_test_subset.iloc[:10])

        # folds 2017 - 2020
        df = df.loc[df["variable"].str.startswith("start_date")].drop("variable", axis=1)
        # print(df)
        # print(df.loc[(df["check"] == 0) & (df["group"] >= 9) , :])

        # training set
        train_subset = df.loc[
            (df["group"] != test_fold) & (df["group"] != val_fold) & (df["check"] == 0),
            ["path", "start_date", "end_date"],
        ].sort_values(by=["path","start_date"])
        # print(train_subset.iloc[:10])

        # validation set
        val_subset = df.loc[
            (df["group"] == val_fold) & (df["check"] == 0),
            ["path", "start_date", "end_date"],
        ].sort_values(by=["path","start_date"])
        # print(val_subset.iloc[:10])

        # iid test set
        spatial_test_subset = df.loc[
            (df["group"] == test_fold) & (df["check"] == 0),
            ["path", "start_date", "end_date"],
        ].sort_values(by=["path","start_date"])
        # print(spatial_test_subset.iloc[:10])

        return train_subset, val_subset, spatial_test_subset, temporal_test_subset

    def dataset_split_continent(self):
        test_fold = self.hparams.test_fold
        val_fold = self.hparams.val_fold

        # Read your CSV file into a pandas DataFrame
        df = pd.read_csv(self.hparams.fold_path, delimiter=",")[
            [
                "path",
                "group",
                "check",
                "start_date",
                "lat",
                "lon",
            ]
        ]

        # 3 training minicubes between 2017 - 2020 (randomly starting in the first year), followed by 3 test minicubes in 2021
        df["start_date"] = pd.to_datetime(
            df["start_date"], format="%Y-%m-%dT%H:%M:%S.%f"
        )
        df["start_date2"] = df["start_date"] + datetime.timedelta(days=450)
        df["start_date3"] = df["start_date2"] + datetime.timedelta(days=450)
        df["start_test1"] = df["start_date3"] + datetime.timedelta(days=450)
        df["start_test2"] = df["start_test1"] + datetime.timedelta(days=90)
        df["start_test3"] = df["start_test2"] + datetime.timedelta(days=90)
        df = df.melt(["path", "group", "check", "lat", "lon"], value_name="start_date")
        df["end_date"] = df["start_date"] + datetime.timedelta(days=449)

        # Check if the point is in Africa
        def is_in_africa(row, africa):
            # Create a Shapely Point from latitude and longitude
            point = Point(row["lon"], row["lat"])

            # Check if the point is in Africa
            return africa.contains(point).any()

        # Load a world map dataset (included in geopandas)
        world = gpd.read_file("https://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_110m_admin_0_countries.geojson")
        africa = world[world["continent"] == "Africa"]
        # Apply the is_in_africa function to each row
        df["in_africa"] = df.apply(lambda row: is_in_africa(row, africa), axis=1)

        # temporal test set 2021
        temporal_test_subset = df.loc[
            (df["variable"].str.startswith("start_test")),
            ["path", "start_date", "end_date"],
        ]

        # folds 2017 - 2020
        df = df.loc[df["variable"].str.startswith("start_date")].drop("variable", 1)

        # training set
        train_subset = df.loc[
            (df["group"] != test_fold)
            & (df["group"] != val_fold)
            & (df["check"] == 0)
            & (df["in_africa"] == False),
            ["path", "start_date", "end_date"],
        ]

        # validation set
        val_subset = df.loc[
            (df["group"] == val_fold) & (df["check"] == 0) & (df["in_africa"] == False),
            ["path", "start_date", "end_date"],
        ]

        # iid test set
        spatial_test_subset = df.loc[
            (df["group"] == test_fold)
            & (df["check"] == 0)
            & (df["in_africa"] == False),
            ["path", "start_date", "end_date"],
        ]

        africa_test_subset = df.loc[
            (df["check"] == 0) & (df["in_africa"] == True),
            ["path", "start_date", "end_date"],
        ]

        return (
            train_subset,
            val_subset,
            spatial_test_subset,
            temporal_test_subset,
            africa_test_subset,
        )
