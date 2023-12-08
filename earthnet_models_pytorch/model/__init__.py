from earthnet_models_pytorch.model.local_rnn import LocalRNN
from earthnet_models_pytorch.model.rnn import RNN
from earthnet_models_pytorch.model.convlstm_ae import ConvLSTMAE
from earthnet_models_pytorch.model.convlstm_former import ConvLSTMAEFormer
from earthnet_models_pytorch.model.simvp import SimVP
from earthnet_models_pytorch.model.predrnn import PredRNN
from earthnet_models_pytorch.model.nextframe_resnet import NextFrameResNet
from earthnet_models_pytorch.model.nextframe_unet import NextFrameUNet
from earthnet_models_pytorch.model.spatiotemporal_unet import SpatioTemporalUNet
from earthnet_models_pytorch.model.unet_lstm import UNetLSTM
from earthnet_models_pytorch.model.contextformer import ContextFormer
from earthnet_models_pytorch.model.presto import Presto

from earthnet_models_pytorch.task import TASKS

MODELS = {
    "local-rnn": LocalRNN,
    "rnn": RNN,
    "convlstm_ae": ConvLSTMAE,
    "context-convlstm": ContextConvLSTM,
    "dumby-mlp": CNN,
    "simvp": SimVP,
    "predrnn": PredRNN,
    "nf_resnet": NextFrameResNet,
    "nf_unet": NextFrameUNet,
    "st_unet": SpatioTemporalUNet,
    "convlstm_ae": ConvLSTMAE,
    "unet_lstm": UNetLSTM,
    "contextformer": ContextFormer,
    "convlstm_former": ConvLSTMAEFormer,
    "presto": Presto
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
    "convlstm_ae": TASKS["spatio-temporal"],
    "contextformer": TASKS["spatio-temporal"],
    "convlstm_former": TASKS["spatio-temporal"],
    "presto": TASKS["spatio-temporal"],
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
    "convlstm_ae": "spatio-temporal",
    "contextformer": "spatio-temporal",
    "convlstm_former": "spatio-temporal",
    "presto": "spatio-temporal",
}
