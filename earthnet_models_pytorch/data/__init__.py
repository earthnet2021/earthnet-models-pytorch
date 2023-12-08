from earthnet_models_pytorch.data.en21_data import EarthNet2021DataModule
from earthnet_models_pytorch.data.en22_data import EarthNet2022DataModule
from earthnet_models_pytorch.data.en21x_data import EarthNet2021XDataModule
from earthnet_models_pytorch.data.en21x_data_old import (
    EarthNet2021XOldDataModule,
    EarthNet2021XpxOldDataModule,
)
from earthnet_models_pytorch.data.en23_data import EarthNet2023DataModule

SETTINGS = [
    "en21-std",
    "en21-veg",
    "europe-veg",
    "en21x",
    "en21xold",
    "en21x-pxold",
    "en22",
    "en23",
]

DATASETS = {
    "en21-std": EarthNet2021DataModule,
    "en21-veg": EarthNet2021DataModule,
    "en21xold": EarthNet2021XOldDataModule,
    "en21x": EarthNet2021XDataModule,
    "en21x-pxold": EarthNet2021XpxOldDataModule,
    "en22": EarthNet2022DataModule,
    "en23": EarthNet2023DataModule,
}
