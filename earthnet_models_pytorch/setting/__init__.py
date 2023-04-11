from earthnet_models_pytorch.setting.en21_data import EarthNet2021DataModule
from earthnet_models_pytorch.setting.en22_data import EarthNet2022DataModule
from earthnet_models_pytorch.setting.en21x_data import EarthNet2021XDataModule
from earthnet_models_pytorch.setting.en21x_data_old import (
    EarthNet2021XOldDataModule,
    EarthNet2021XpxOldDataModule,
)
from earthnet_models_pytorch.setting.en23_data import EarthNet2023DataModule

from earthnet_models_pytorch.setting.en21_std_metric import EarthNetScore
from earthnet_models_pytorch.setting.en21_veg_metric import (
    RootMeanSquaredError as RMSEVegScore,
)
from earthnet_models_pytorch.setting.en21_veg_metric import (
    RMSE_ens21x,
    RMSE_ens22,
    RMSE_ens23,
)
from earthnet_models_pytorch.setting.nnse_metric import NNSEVeg

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
METRICS = {
    "en21-std": EarthNetScore,
    "en21-veg": RMSEVegScore,
    "en21x": NNSEVeg,
    "en21xold": RMSE_ens21x,
    "en21x-pxold": RMSE_ens21x,
    "en22": RMSE_ens22,
    "en23": RMSE_ens23,
}
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
        "monitor": "veg_score",
        "filename": "Epoch-{epoch:02d}-veg_score-{veg_score:.4f}",
        "mode": "max",
    },
}
