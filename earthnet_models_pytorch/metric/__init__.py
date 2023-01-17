from earthnet_models_pytorch.metric.EarthNetScore_metric import EarthNetScore
from earthnet_models_pytorch.metric.RMSE_metric import RootMeanSquaredError, RMSE_ens22
from earthnet_models_pytorch.metric.NNSE_metric import NormalizedNashSutcliffeEfficiency


METRICS = {
    "en21-std": EarthNetScore,
    "en21-veg": RootMeanSquaredError,
    "en21x": RootMeanSquaredError,
    "en21x-px": RootMeanSquaredError,
    "en22": RMSE_ens22,
    "en23": NormalizedNashSutcliffeEfficiency,
}
