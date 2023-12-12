from typing import Union, Optional

import argparse
import sys
import copy
import multiprocessing
import re
import json
from pathlib import Path
import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr
from torch.utils.data import Dataset, DataLoader, random_split

from earthnet_models_pytorch.utils import str2bool

variables = {
    # Sentinel 2 bands. Spatio-temporal data. Every 5 days.
    "s2_bands": ["s2_B02", "s2_B03", "s2_B04", "s2_B05", "s2_B06", "s2_B07", "s2_B8A"],
    "s2_avail": ["s2_avail"],
    "s2_scene_classification": ["s2_SCL"],
    "cloud_mask": ["s2_mask"],
    # Sentinel 1 bands. Spatio-temporal data. Every 12 days.
    "s1_bands": ["s1_vv", "s1_vh"],
    "s1_avail": ["s1_avail"],
    # Monthly mean and std computed over 1984-2020 using Landsat 5, 7, 8. Digital Earth Africa product.
    "ndviclim": ["ndviclim_mean", "ndviclim_std"],
    # Era5 reanalysis dataset. Temporal dataset. Era5 era5land pev and pet have a low precision.
    "era5lands": [
        "era5land_t2m_mean",
        "era5land_pev_mean",
        "era5land_slhf_mean",
        "era5land_ssr_mean",
        "era5land_sp_mean",
        "era5land_sshf_mean",
        "era5land_e_mean",
        "era5land_tp_mean",
        "era5land_t2m_min",
        "era5land_pev_min",
        "era5land_slhf_min",
        "era5land_ssr_min",
        "era5land_sp_min",
        "era5land_sshf_min",
        "era5land_e_min",
        "era5land_tp_min",
        "era5land_t2m_max",
        "era5land_pev_max",
        "era5land_slhf_max",
        "era5land_ssr_max",
        "era5land_sp_max",
        "era5land_sshf_max",
        "era5land_e_max",
        "era5land_tp_max",
    ],
    # Era5 recomputed by Fabian Gans for the XAIDA project. More accurate.
    "era5": [
        "era5_e",
        "era5_pet",
        "era5_pev",
        "era5_ssrd",
        "era5_t2m",
        "era5_t2mmax",
        "era5_t2mmin",
        "era5_tp",
    ],
    # soigrids is a soil properties dataset. Spatial data.
    "soilgrids": [
        "sg_bdod_top_mean",
        "sg_bdod_sub_mean",
        "sg_cec_top_mean",
        "sg_cec_sub_mean",
        "sg_cfvo_top_mean",
        "sg_cfvo_sub_mean",
        "sg_clay_top_mean",
        "sg_clay_sub_mean",
        "sg_nitrogen_top_mean",
        "sg_nitrogen_sub_mean",
        "sg_phh2o_top_mean",
        "sg_phh2o_sub_mean",
        "sg_ocd_top_mean",
        "sg_ocd_sub_mean",
        "sg_sand_top_mean",
        "sg_sand_sub_mean",
        "sg_silt_top_mean",
        "sg_silt_sub_mean",
        "sg_soc_top_mean",
        "sg_soc_sub_mean",
    ],
    # Elevation map. Defined on 0 - 2000.
    "elevation": ["cop_dem"],  # ['srtm_dem', 'alos_dem', 'cop_dem'],
    # Landscape classes. Categorical variable.
    "landcover": ["esawc_lc"],
    # Geomorphon classes. Categorical variable. geomorphon is not define on 128 x 128 pixels neither.
    "geomorphon": ["geom_cls"],
}

statistic = {"era5land_t2m_mean": {"mean": 294.36909650500024, "min": 262.5877380371094, "max": 315.4031677246094}, "era5land_pev_mean": {"mean": -0.005758406223293537, "min": -0.06333371996879578, "max": 0.007520616985857487}, "era5land_slhf_mean": {"mean": -2779231.7898821747, "min": -32735200.0, "max": 783463.0}, "era5land_ssr_mean": {"mean": 10316129.186564116, "min": 446041.5, "max": 20328616.0}, "era5land_sp_mean": {"mean": 91693.94739265762, "min": 69813.390625, "max": 103730.265625}, "era5land_sshf_mean": {"mean": -3108097.395623107, "min": -13288390.0, "max": 31794404.0}, "era5land_e_mean": {"mean": -0.0011113149884417407, "min": -0.013089891523122787, "max": 0.000274601683486253}, "era5land_tp_mean": {"mean": 0.001274658704506941, "min": 0.0, "max": 0.24756526947021484}, "era5land_t2m_min": {"mean": 289.48678785611975, "min": 252.41452026367188, "max": 309.0274963378906}, "era5land_pev_min": {"mean": -0.010728926082279223, "min": -0.11989240348339081, "max": 2.9400456696748734e-05}, "era5land_slhf_min": {"mean": -4879028.720334868, "min": -61155960.0, "max": 246351.75}, "era5land_ssr_min": {"mean": 2945.1830685694185, "min": 0.0, "max": 3412364.0}, "era5land_sp_min": {"mean": 91493.93692969388, "min": 69627.25, "max": 103637.9765625}, "era5land_sshf_min": {"mean": -5695757.078598437, "min": -21713694.0, "max": 9758987.0}, "era5land_e_min": {"mean": -0.0019509420153387682, "min": -0.02445455826818943, "max": 8.70322110131383e-05}, "era5land_tp_min": {"mean": 0.0001631960775503857, "min": 0.0, "max": 0.0867575854063034}, "era5land_t2m_max": {"mean": 299.9465557238053, "min": 264.7709655761719, "max": 322.0445861816406}, "era5land_pev_max": {"mean": -0.00019507679501445417, "min": -0.017947837710380554, "max": 0.016586333513259888}, "era5land_slhf_max": {"mean": -55752.188274930275, "min": -9668804.0, "max": 1737824.0}, "era5land_ssr_max": {"mean": 17422424.18667931, "min": 807063.0, "max": 32543060.0}, "era5land_sp_max": {"mean": 91881.56694081018, "min": 69905.3671875, "max": 103880.8359375}, "era5land_sshf_max": {"mean": 243473.68240272274, "min": -1445534.0, "max": 60178688.0}, "era5land_e_max": {"mean": -2.2295505859743446e-05, "min": -0.003866284154355526, "max": 0.0005941614508628845}, "era5land_tp_max": {"mean": 0.003298523462095321, "min": 0.0, "max": 0.42113596200942993}, "era5_e": {"mean": 0.0, "min": 0.0, "max": 0.0}, "era5_pet": {"mean": -3.058510107010669, "min": -14.473007202148438, "max": -0.0757131576538086}, "era5_pev": {"mean": -4.68730742862229, "min": -14.496773719787598, "max": 0.29635077714920044}, "era5_ssrd": {"mean": 0.0, "min": 0.0, "max": 0.0}, "era5_t2m": {"mean": 294.7884567954966, "min": 263.9975280761719, "max": 316.1473693847656}, "era5_t2mmax": {"mean": 300.6563908273207, "min": 266.8150329589844, "max": 323.0835876464844}, "era5_t2mmin": {"mean": 289.6629521021342, "min": 255.50045776367188, "max": 310.5412292480469}, "era5_tp": {"mean": 2.3729672696519954, "min": 0.0, "max": 376.09942626953125}, "sg_bdod_top_mean": {"mean": 135.45496200764276, "min": 63.52088165283203, "max": 175.99490356445312}, "sg_bdod_sub_mean": {"mean": 141.8806978813852, "min": 62.23273468017578, "max": 183.96360778808594}, "sg_cec_top_mean": {"mean": 170.49680537986143, "min": 21.397911071777344, "max": 726.3426513671875}, "sg_cec_sub_mean": {"mean": 164.94571095303948, "min": 12.284384727478027, "max": 757.5006103515625}, "sg_cfvo_top_mean": {"mean": 114.48523525434305, "min": 0.3873753547668457, "max": 702.10693359375}, "sg_cfvo_sub_mean": {"mean": 159.2997046874758, "min": 1.4210796356201172, "max": 641.9326171875}, "sg_clay_top_mean": {"mean": 271.71965366429015, "min": 36.165184020996094, "max": 647.458740234375}, "sg_clay_sub_mean": {"mean": 312.52538792582385, "min": 35.33229446411133, "max": 758.8634033203125}, "sg_nitrogen_top_mean": {"mean": 132.20457388449356, "min": 22.74068260192871, "max": 1097.6943359375}, "sg_nitrogen_sub_mean": {"mean": 63.62704720349757, "min": 9.178686141967773, "max": 1728.8538818359375}, "sg_phh2o_top_mean": {"mean": 66.89364659457219, "min": 41.45420837402344, "max": 95.34241485595703}, "sg_phh2o_sub_mean": {"mean": 68.3110269067554, "min": 44.28192138671875, "max": 94.99882507324219}, "sg_ocd_top_mean": {"mean": 195.3052326225848, "min": 60.40400314331055, "max": 625.5231323242188}, "sg_ocd_sub_mean": {"mean": 78.06307836188793, "min": 27.449851989746094, "max": 498.01470947265625}, "sg_sand_top_mean": {"mean": 502.18085207055515, "min": 34.42234420776367, "max": 922.1285400390625}, "sg_sand_sub_mean": {"mean": 471.3682585797815, "min": 18.76363754272461, "max": 925.4884643554688}, "sg_silt_top_mean": {"mean": 226.08421234511314, "min": 12.748359680175781, "max": 635.8508911132812}, "sg_silt_sub_mean": {"mean": 216.10676003610075, "min": 12.029265403747559, "max": 564.3701171875}, "sg_soc_top_mean": {"mean": 166.41441301984628, "min": 17.17481231689453, "max": 1209.2373046875}, "sg_soc_sub_mean": {"mean": 73.5223076305796, "min": 9.3042631149292, "max": 1012.6757202148438}}


class EarthNet2023Dataset(Dataset):
    def __init__(
        self, folder: Union[Path, str], target: str, variables=variables, fp16=False
    ):
        if not isinstance(folder, Path):
            folder = Path(folder)
        self.filepaths = (
            sorted(list(folder.glob("*.nc")))
            if len(sorted(list(folder.glob("*.nc")))) > 0
            else sorted(list(folder.glob("*/*.nc")))
        )
        print(folder)
        self.type = np.float16 if fp16 else np.float32
        self.target = target
        self.variables = variables

    def __getitem__(self, idx: int) -> dict:
        # Open the minicube
        filepath = self.filepaths[idx]
        minicube = xr.open_dataset(filepath)

        # Select the days with available data
        indexes_avail = np.where(minicube.s2_avail.values == 1)[0]
        # s2 is every 5 days
        time = [minicube.time.values[i] for i in range(4, 450, 5)]
        dates = [minicube.time.values[i] for i in (indexes_avail)]

        # Condition to check that the every 5 days inclus all the dates available (+ missing day)
        if not set(dates) <= set(time):
            raise AssertionError(
                "ERROR: time indexes of the minicubes are not consistant ", filepath
            )

        # Create the minicube
        # Sentinel-2: s2 is 10 to 5 days, and already rescaled [0, 1]
        s2_cube = (
            minicube[self.variables["s2_bands"]]
            .to_array()
            .sel(time=time)
            .values.transpose((1, 0, 2, 3))
            .astype(self.type)
        )  # shape: (time, channels, w, h)

        s2_mask = (
            minicube[self.variables["cloud_mask"]]
            .to_array()
            .sel(time=time)
            .values.transpose((1, 0, 2, 3))
            .astype(self.type)
        )  # (time, 1, w, h)

        target = (
            self.target_computation(minicube)
            .sel(time=time)
            .values[:, None, ...]
            .astype(self.type)
        )

        topography = (
            minicube[self.variables["elevation"]].to_array().values.astype(self.type)
            / 2000
        )  # c h w,

        landcover = (
            minicube[self.variables["landcover"]].to_array().values.astype(self.type)
        )  # c h w

        # Rescaling

        for variable in (
            self.variables["era5"]
            + self.variables["era5lands"]
            + self.variables["soilgrids"]
        ):
            minicube[variable] = (minicube[variable] - statistic[variable]["min"]) / (
                statistic[variable]["max"] - statistic[variable]["min"]
            )

        # Era5land and Era5 dataset. Weather is daily
        meteo_cube = (
            minicube[self.variables["era5"]]  # + self.variables["era5lands"]]
            .to_array()
            .values.transpose((1, 0))
            .astype(self.type)
        )

        # Soilgrid dataset
        sg_cube = (
            minicube[self.variables["soilgrids"]].to_array().values.astype(self.type)
        )

        # NaN values handling
        s2_cube = np.where(np.isnan(s2_cube), np.zeros(1).astype(self.type), s2_cube)
        s2_mask = np.where(np.isnan(s2_mask), np.ones(1).astype(self.type), s2_cube)
        target = np.where(np.isnan(target), np.zeros(1).astype(self.type), target)
        meteo_cube = np.where(
            np.isnan(meteo_cube), np.zeros(1).astype(self.type), meteo_cube
        )
        sg_cube = np.where(np.isnan(sg_cube), np.zeros(1).astype(self.type), sg_cube)
        topography = np.where(
            np.isnan(topography), np.zeros(1).astype(self.type), topography
        )
        landcover = np.where(
            np.isnan(landcover), np.zeros(1).astype(self.type), landcover
        )

        # Concatenation
        satellite_data = np.concatenate((target, s2_cube), axis=1)

        # Final minicube
        data = {
            "dynamic": [
                torch.from_numpy(satellite_data),
                torch.from_numpy(meteo_cube),
            ],
            "dynamic_mask": [torch.from_numpy(s2_mask)],
            "static": [torch.from_numpy(topography), torch.from_numpy(sg_cube)],
            # "target": torch.from_numpy(target),
            "landcover": torch.from_numpy(landcover),
            "filepath": str(filepath),
            "cubename": self.__name_getter(filepath),
        }

        return data

    def __len__(self) -> int:
        return len(self.filepaths)

    def __name_getter(self, path: Path) -> str:
        """
        Helper function gets Cubename from a Path
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

    def target_computation(self, minicube) -> str:
        """Compute the vegetation index (VI) target"""
        if self.target == "ndvi":
            targ = (minicube.s2_B8A - minicube.s2_B04) / (
                minicube.s2_B8A + minicube.s2_B04 + 1e-6
            )

        if (
            self.target == "kndvi"
        ):  # TODO the denominator is not optimal, needs to be improved accordingly to the original paper
            targ = np.tanh(
                (
                    (minicube.s2_B08 - minicube.s2_B04)
                    / (minicube.s2_BO8 + minicube.s2_B04 + 1e-6)
                )
                ** 2
            ) / np.tanh(1)

        if self.target == "anomalie_ndvi":
            targ = (
                (minicube.s2_B8A - minicube.s2_B04)
                / (minicube.s2_B8A + minicube.s2_B04 + 1e-6)
            ) - minicube.msc

        return targ


class EarthNet2023DataModule(pl.LightningDataModule):
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
            earthnet_full = EarthNet2023Dataset(
                self.base_dir / "train",
                target=self.hparams.target,
                fp16=self.hparams.fp16,
            )
            self.earthnet_train, self.earthnet_val = random_split(
                earthnet_full,
                [0.95, 0.05],
                generator=torch.Generator().manual_seed(self.hparams.val_split_seed),
            )

        if stage == "test" or stage is None:
            self.earthnet_test = EarthNet2023Dataset(
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
