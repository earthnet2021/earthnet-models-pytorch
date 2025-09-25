from earthnet_models_pytorch.model.local_rnn import LocalRNN
from earthnet_models_pytorch.model.rnn import RNN
from earthnet_models_pytorch.model.convlstm_ae import ConvLSTMAE
from earthnet_models_pytorch.model.simvp import SimVP
from earthnet_models_pytorch.model.predrnn import PredRNN
from earthnet_models_pytorch.model.nextframe_resnet import NextFrameResNet
from earthnet_models_pytorch.model.nextframe_unet import NextFrameUNet
from earthnet_models_pytorch.model.spatiotemporal_unet import SpatioTemporalUNet
from earthnet_models_pytorch.model.unet_lstm import UNetLSTM
from earthnet_models_pytorch.model.contextformer import ContextFormer

MODELS = {
    "local-rnn": LocalRNN,
    "rnn": RNN,
    "convlstm_ae": ConvLSTMAE,
    "simvp": SimVP,
    "predrnn": PredRNN,
    "nf_resnet": NextFrameResNet,
    "nf_unet": NextFrameUNet,
    "st_unet": SpatioTemporalUNet,
    "convlstm_ae": ConvLSTMAE,
    "unet_lstm": UNetLSTM,
    "contextformer": ContextFormer,
}

