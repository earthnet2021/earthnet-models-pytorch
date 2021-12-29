

from earthnet_models_pytorch.model.channel_u_net import ChannelUNet
from earthnet_models_pytorch.model.local_rnn import LocalRNN
from earthnet_models_pytorch.model.rnn import RNN
from earthnet_models_pytorch.task import TASKS
from earthnet_models_pytorch.model.codyn.codyn_body import CodynBody

MODELS = {
    "channel-u-net": ChannelUNet,
    "local-rnn": LocalRNN,
    "rnn": RNN,
    "codyn": CodynBody
}

MODELTASKS = {
    "channel-u-net": TASKS["spatio-temporal"],
    "local-rnn": TASKS["spatio-temporal"],
    "rnn": TASKS["spatio-temporal"],
    "codyn": TASKS["spatio-temporal"]
}

MODELTASKNAMES = {
    "channel-u-net": "spatio-temporal",
    "local-rnn": "spatio-temporal",
    "rnn": "spatio-temporal",
    "codyn": "spatio-temporal"
}