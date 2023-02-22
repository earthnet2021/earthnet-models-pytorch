
from earthnet_models_pytorch.model.local_rnn import LocalRNN
from earthnet_models_pytorch.model.rnn import RNN
from earthnet_models_pytorch.model.context_convlstm import ContextConvLSTM
from earthnet_models_pytorch.model.CNN import CNN
from earthnet_models_pytorch.model.convlstm_lstm import ConvLSTMLSTM
from earthnet_models_pytorch.model.simvp import SimVP
from earthnet_models_pytorch.model.predrnn import PredRNN
from earthnet_models_pytorch.model.convlstm import ConvLSTM
from earthnet_models_pytorch.model.nextframe_resnet import NextFrameResNet
from earthnet_models_pytorch.model.nextframe_unet import NextFrameUNet
from earthnet_models_pytorch.model.spatiotemporal_unet import SpatioTemporalUNet
from earthnet_models_pytorch.model.unet_lstm import UNetLSTM
from earthnet_models_pytorch.task import TASKS

MODELS = {
    "local-rnn": LocalRNN,
    "rnn": RNN,
    "context-convlstm": ContextConvLSTM,
    "dumby-mlp": CNN,
    "convlstm-lstm": ConvLSTMLSTM,
    "simvp": SimVP,
    "predrnn": PredRNN,
    "convlstm": ConvLSTM,
    "nf_resnet": NextFrameResNet,
    "nf_unet": NextFrameUNet,
    "st_unet": SpatioTemporalUNet,
    "unet_lstm": UNetLSTM
}

MODELTASKS = {
    "channel-u-net": TASKS["spatio-temporal"],
    "local-rnn": TASKS["spatio-temporal"],
    "rnn": TASKS["spatio-temporal"],
    "context-convlstm": TASKS["spatio-temporal"],
    "u-net-convlstm": TASKS["spatio-temporal"],
    "dumby-mlp": TASKS["spatio-temporal"],
    'convlstm-lstm': TASKS["spatio-temporal"],
    'simvp': TASKS["spatio-temporal"],
    "predrnn":  TASKS["spatio-temporal"],
    "nf_resnet":  TASKS["spatio-temporal"],
    "convlstm": TASKS["spatio-temporal"],
    "nf_unet": TASKS["spatio-temporal"],
    "st_unet": TASKS["spatio-temporal"],
    "unet_lstm": TASKS["spatio-temporal"],
}

MODELTASKNAMES = {
    "channel-u-net": "spatio-temporal",
    "local-rnn": "spatio-temporal",
    "rnn": "spatio-temporal",
    "context-convlstm": "spatio-temporal",
    "u-net-convlstm" : "spatio-temporal",
    "dumby-mlp" : "spatio-temporal",
    "convlstm-lstm": "spatio-temporal",
    "simvp": "spatio-temporal",
    "predrnn": "spatio-temporal",
    "nf_resnet": "spatio-temporal",
    "convlstm": "spatio-temporal",
    "nf_unet": "spatio-temporal",
    "st_unet": "spatio-temporal",
    "unet_lstm": "spatio-temporal",
}