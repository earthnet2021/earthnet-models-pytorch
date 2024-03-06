"""EarthNet Models PyTorch
A library with models for Earth surface forecasting.
"""
__version__ = "0.1.0"
__author__ = 'Vitus Benson, Claire Robin'
__credits__ = 'Max Planck Institute for Biogeochemistry'

from earthnet_models_pytorch.utils import str2bool
import earthnet_models_pytorch.data
import earthnet_models_pytorch.metric
import earthnet_models_pytorch.model
import earthnet_models_pytorch.task
import earthnet_models_pytorch.utils