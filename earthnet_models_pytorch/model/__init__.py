

from lib2to3.pytree import Base
from earthnet_models_pytorch.model.channel_u_net import ChannelUNet
from earthnet_models_pytorch.model.local_rnn import LocalRNN
from earthnet_models_pytorch.model.rnn import RNN
from earthnet_models_pytorch.model.context_convlstm import ContextConvLSTM
from earthnet_models_pytorch.model.baseline import Baseline
from earthnet_models_pytorch.task import TASKS

MODELS = {
    "channel-u-net": ChannelUNet,
    "local-rnn": LocalRNN,
    "rnn": RNN,
    "context-convlstm": ContextConvLSTM,
    "baseline": Baseline
}

MODELTASKS = {
    "channel-u-net": TASKS["spatio-temporal"],
    "local-rnn": TASKS["spatio-temporal"],
    "rnn": TASKS["spatio-temporal"],
    "context-convlstm": TASKS["spatio-temporal"],
    "baseline": TASKS["spatio-temporal"]
}

MODELTASKNAMES = {
    "channel-u-net": "spatio-temporal",
    "local-rnn": "spatio-temporal",
    "rnn": "spatio-temporal",
    "context-convlstm": "spatio-temporal",
    "baseline": "spatio-temporal"
}