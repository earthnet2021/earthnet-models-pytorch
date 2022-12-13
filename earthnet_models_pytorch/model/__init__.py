from earthnet_models_pytorch.model.local_rnn import LocalRNN
from earthnet_models_pytorch.model.rnn import RNN
from earthnet_models_pytorch.model.context_convlstm import ContextConvLSTM
from earthnet_models_pytorch.model.cnn import CNN
from earthnet_models_pytorch.model.convlstm_lstm import ConvLSTMLSTM
from earthnet_models_pytorch.model.convlstm_ae import ConvLSTMAE
# from earthnet_models_pytorch.task import TASKS

MODELS = {
    "local-rnn": LocalRNN,
    "rnn": RNN,
    "context-convlstm": ContextConvLSTM,
    "cnn": CNN,
    "convlstm-lstm": ConvLSTMLSTM,
    "convlstm_ae": ConvLSTMAE
}

# MODELTASKS = {
#    "channel-u-net": TASKS["spatio-temporal"],
#    "local-rnn": TASKS["spatio-temporal"],
#    "rnn": TASKS["spatio-temporal"],
#    "context-convlstm": TASKS["spatio-temporal"],
#    "u-net-convlstm": TASKS["spatio-temporal"],
#    "dumby-mlp": TASKS["spatio-temporal"],
#    'convlstm-lstm': TASKS["spatio-temporal"]
#}

# MODELTASKNAMES = {
#     "channel-u-net": "spatio-temporal",
#     "local-rnn": "spatio-temporal",
#     "rnn": "spatio-temporal",
#     "context-convlstm": "spatio-temporal",
#     "u-net-convlstm" : "spatio-temporal",
#     "dumby-mlp" : "spatio-temporal",
#     "convlstm-lstm": "spatio-temporal"
# }