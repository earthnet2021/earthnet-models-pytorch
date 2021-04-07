

from earthnet_models_pytorch.model.channel_u_net import ChannelUNet
from earthnet_models_pytorch.task import TASKS

MODELS = {
    "channel-u-net": ChannelUNet
}

MODELTASKS = {
    "channel-u-net": TASKS["spatio-temporal"]
}

MODELTASKNAMES = {
    "channel-u-net": "spatio-temporal"
}