from earthnet_models_pytorch.model import LocalRNN, RNN, ConvLSTM, CNN, ConvLSTMLSTM
# from earthnet_models_pytorch.task import TASKS

MODELS = {
    "local-rnn": LocalRNN,
    "rnn": RNN,
    "context-convlstm": ConvLSTM,
    "cnn": CNN,
    "convlstm-lstm": ConvLSTMLSTM
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