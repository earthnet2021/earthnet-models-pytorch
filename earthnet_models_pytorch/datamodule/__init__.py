

from earthnet_models_pytorch.datamodule.en21_data import EarthNet2021DataModule, EarthNet2021Dataset
from earthnet_models_pytorch.datamodule.en22_data import EarthNet2022DataModule, EarthNet2022Dataset
from earthnet_models_pytorch.datamodule.en23_data import EarthNet2023DataModule, EarthNet2023Dataset
from earthnet_models_pytorch.datamodule.en21x_data import EarthNet2021XDataModule, EarthNet2021XpxDataModule, EarthNet2021XDataset, EarthNet2021XpxDataset

DATAMODULES = {"en21-std": EarthNet2021DataModule, "en21-veg": EarthNet2021DataModule, "en21x": EarthNet2021XDataModule, 
            "en21x-px": EarthNet2021XpxDataModule, "en22": EarthNet2022DataModule, "en23": EarthNet2023DataModule}
DATASETS = {"en21-std": EarthNet2021Dataset, "en21-veg": EarthNet2021Dataset, "en21x": EarthNet2021XDataset, 
            "en21x-px": EarthNet2021XpxDataset, "en22": EarthNet2022Dataset, "en23": EarthNet2023Dataset}

METRIC_CHECKPOINT_INFO = {  # to refactor too from METRICS? maybe remove mode too
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
        },
    "en22": {
        "monitor": "RMSE_Veg",
        "filename": 'Epoch-{epoch:02d}-RMSE (Vegetation)-{RMSE_Veg:.4f}',
        "mode": 'min'
        },
    "en23": {
        "monitor": "Veg_score",
        "filename": 'Epoch-{epoch:02d}-Veg_score (Vegetation)-{NNSE_Veg:.4f}',
        "mode": 'min'
        }
}
