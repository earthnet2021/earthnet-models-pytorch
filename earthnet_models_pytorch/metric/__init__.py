from earthnet_models_pytorch.metric.EarthNetScore_metric import EarthNetScore
from earthnet_models_pytorch.metric.RMSE_metric import RootMeanSquaredError
from earthnet_models_pytorch.metric.NNSE_metric import NormalizedNashSutcliffeEfficiency

METRICS = {"ENS": EarthNetScore, "RMSE": RootMeanSquaredError, "NNSE": NormalizedNashSutcliffeEfficiency}
