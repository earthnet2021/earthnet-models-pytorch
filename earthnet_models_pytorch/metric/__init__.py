from earthnet_models_pytorch.metric.earthnetscore import EarthNetScore
from earthnet_models_pytorch.metric.rmse import RootMeanSquaredError
from earthnet_models_pytorch.metric.nnse import NormalizedNashSutcliffeEfficiency

METRICS = {"ENS": EarthNetScore, "RMSE": RootMeanSquaredError, "NNSE": NormalizedNashSutcliffeEfficiency}