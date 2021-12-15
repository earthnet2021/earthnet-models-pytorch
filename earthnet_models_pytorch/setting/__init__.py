

from earthnet_models_pytorch.setting.en21_data import EarthNet2021DataModule
from earthnet_models_pytorch.setting.en21x_data import EarthNet2021XDataModule,EarthNet2021XpxDataModule
from earthnet_models_pytorch.setting.en21_std_metric import EarthNetScore
from earthnet_models_pytorch.setting.en21_veg_metric import RootMeanSquaredError as RMSEVegScore
from earthnet_models_pytorch.setting.en21_veg_metric import RMSE_ens21x

SETTINGS = ["en21-std", "en21-veg", "europe-veg", "en21x","en21x-px"]
METRICS = {"en21-std": EarthNetScore, "en21-veg": RMSEVegScore, "en21x": RMSE_ens21x,"en21x-px": RMSE_ens21x}
DATASETS = {"en21-std": EarthNet2021DataModule, "en21-veg": EarthNet2021DataModule, "en21x": EarthNet2021XDataModule, "en21x-px": EarthNet2021XpxDataModule}
METRIC_CHECKPOINT_INFO = {
    "en21-std": {
        "monitor": "EarthNetScore",
        "filename": 'Epoch-{epoch:02d}-ENS-{EarthNetScore:.4f}',
        "mode": 'max'
    },
    "en21-veg": {
        "monitor": "RMSE_Veg",
        "filename": 'Epoch-{epoch:02d}-RMSE (Vegetation)-{RMSE_Veg:.4f}',
        "mode": 'min'
        },
    "en21x": {
        "monitor": "RMSE_Veg",
        "filename": 'Epoch-{epoch:02d}-RMSE (Vegetation)-{RMSE_Veg:.4f}',
        "mode": 'min'
        },
    "en21x-px": {
        "monitor": "RMSE_Veg",
        "filename": 'Epoch-{epoch:02d}-RMSE (Vegetation)-{RMSE_Veg:.4f}',
        "mode": 'min'
        }
}
