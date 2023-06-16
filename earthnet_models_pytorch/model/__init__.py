from earthnet_models_pytorch.model.local_rnn import LocalRNN
from earthnet_models_pytorch.model.rnn import RNN
from earthnet_models_pytorch.model.convlstm_ae import ConvLSTMAE
from earthnet_models_pytorch.model.convlstm_lstm import ConvLSTMLSTM
from earthnet_models_pytorch.model.simvp import SimVP
from earthnet_models_pytorch.model.predrnn import PredRNN
from earthnet_models_pytorch.model.convlstm import ConvLSTM
from earthnet_models_pytorch.model.nextframe_resnet import NextFrameResNet

MODELS = {
    "local-rnn": LocalRNN,
    "rnn": RNN,
    "convlstm_ae": ConvLSTMAE,
    "convlstm-lstm": ConvLSTMLSTM,
    "simvp": SimVP,
    "predrnn": PredRNN,
    "convlstm": ConvLSTM,
    "nf_resnet": NextFrameResNet,
}

