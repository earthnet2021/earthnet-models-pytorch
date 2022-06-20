

from lib2to3.pytree import Base
from earthnet_models_pytorch.model.channel_u_net import ChannelUNet
from earthnet_models_pytorch.model.local_rnn import LocalRNN
from earthnet_models_pytorch.model.rnn import RNN
from earthnet_models_pytorch.model.context_convlstm import ContextConvLSTM
from earthnet_models_pytorch.model.u_net_convlstm import UnetConvLSTM
from earthnet_models_pytorch.model.dumby_mlp import DumbyMLP
from earthnet_models_pytorch.model.convlstm_lstm import ConvLSTMLSTM
from earthnet_models_pytorch.task import TASKS

MODELS = {
    "channel-u-net": ChannelUNet,
    "local-rnn": LocalRNN,
    "rnn": RNN,
    "context-convlstm": ContextConvLSTM,
    "u-net-convlstm": UnetConvLSTM,
    "dumby-mlp": DumbyMLP,
    "convlstm-lstm": ConvLSTMLSTM
}

MODELTASKS = {
    "channel-u-net": TASKS["spatio-temporal"],
    "local-rnn": TASKS["spatio-temporal"],
    "rnn": TASKS["spatio-temporal"],
    "context-convlstm": TASKS["spatio-temporal"],
    "u-net-convlstm": TASKS["spatio-temporal"],
    "dumby-mlp": TASKS["spatio-temporal"],
    'convlstm-lstm': TASKS["spatio-temporal"]
}

MODELTASKNAMES = {
    "channel-u-net": "spatio-temporal",
    "local-rnn": "spatio-temporal",
    "rnn": "spatio-temporal",
    "context-convlstm": "spatio-temporal",
    "u-net-convlstm" : "spatio-temporal",
    "dumby-mlp" : "spatio-temporal",
    "convlstm-lstm": "spatio-temporal"
}