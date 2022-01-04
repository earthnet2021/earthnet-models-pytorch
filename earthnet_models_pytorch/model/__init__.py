

from earthnet_models_pytorch.model.channel_u_net import ChannelUNet
from earthnet_models_pytorch.model.local_rnn import LocalRNN
from earthnet_models_pytorch.model.rnn import RNN
from earthnet_models_pytorch.task import TASKS
from earthnet_models_pytorch.model.codyn.codyn_body import CodynBody
from earthnet_models_pytorch.model.hybrid_gsi import HybridGSI

MODELS = {
    "channel-u-net": ChannelUNet,
    "local-rnn": LocalRNN,
    "rnn": RNN,
    "codyn": CodynBody,
    "hybrid-gsi": HybridGSI
}

MODELTASKS = {
    "channel-u-net": TASKS["spatio-temporal"],
    "local-rnn": TASKS["spatio-temporal"],
    "rnn": TASKS["spatio-temporal"],
    "codyn": TASKS["spatio-temporal"],
    "hybrid-gsi": TASKS["spatio-temporal"]
}

MODELTASKNAMES = {
    "channel-u-net": "spatio-temporal",
    "local-rnn": "spatio-temporal",
    "rnn": "spatio-temporal",
    "codyn": "spatio-temporal",
    "hybrid-gsi": "spatio-temporal"
}