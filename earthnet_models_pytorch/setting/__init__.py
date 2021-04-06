

from earthnet_models_pytorch.setting.en21_data import EarthNet2021DataModule
from earthnet_models_pytorch.setting.en21_std_metric import EarthNetScore
from earthnet_models_pytorch.setting.en21_veg_metric import RootMeanSquaredError as RMSEVegScore

SETTINGS = ["en21-std", "en21-veg", "europe-veg"]
METRICS = {"en21-std": EarthNetScore, "en21-veg": RMSEVegScore}
DATASETS = {"en21-std": EarthNet2021DataModule, "en21-veg": EarthNet2021DataModule}
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
        }
}
