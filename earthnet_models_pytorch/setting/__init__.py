from earthnet_models_pytorch.datamodule.en21_data import EarthNet2021DataModule
from earthnet_models_pytorch.datamodule.en22_data import EarthNet2022DataModule
from earthnet_models_pytorch.datamodule.en21x_data import EarthNet2021XDataModule
from earthnet_models_pytorch.datamodule.en21x_data_old import (
    EarthNet2021XOldDataModule,
    EarthNet2021XpxOldDataModule,
)
from earthnet_models_pytorch.datamodule.en23_data import EarthNet2023DataModule

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

METRIC_CHECKPOINT_INFO = {
    "en21-std": {
        "monitor": "EarthNetScore",
        "filename": "Epoch-{epoch:02d}-ENS-{EarthNetScore:.4f}",
        "mode": "max",
    },
    "en21-veg": {
        "monitor": "RMSE_Veg",
        "filename": "Epoch-{epoch:02d}-RMSE (Vegetation)-{RMSE_Veg:.4f}",
        "mode": "min",
    },
    "en21x": {
        "monitor": "veg_score",
        "filename": "Epoch-{epoch:02d}-veg_score-{veg_score:.4f}",
        "mode": "max",
    },
    "en21xold": {
        "monitor": "RMSE_Veg",
        "filename": "Epoch-{epoch:02d}-RMSE (Vegetation)-{RMSE_Veg:.4f}",
        "mode": "min",
    },
    "en21x-pxold": {
        "monitor": "RMSE_Veg",
        "filename": "Epoch-{epoch:02d}-RMSE (Vegetation)-{RMSE_Veg:.4f}",
        "mode": "min",
    },
    "en22": {
        "monitor": "RMSE_Veg",
        "filename": "Epoch-{epoch:02d}-RMSE (Vegetation)-{RMSE_Veg:.4f}",
        "mode": "min",
    },
    "en23": {
        "monitor": "RMSE_Veg",
        "filename": "Epoch-{epoch:02d}-veg_score-{veg_score:.4f}",
        "mode": "max",
    },
}
