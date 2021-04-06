


from earthnet_models_pytorch.task.loss import setup_loss
from earthnet_models_pytorch.task.shedule import SHEDULERS
from earthnet_models_pytorch.task.spatio_temporal import SpatioTemporalTask

TASKS = {
    "spatio-temporal": SpatioTemporalTask
}


TRACK_INFO = {
    "spatio-temporal": {
        "iid": {
            "context_length": 10,
            "target_length": 20
        },
        "ood": {
            "context_length": 10,
            "target_length": 20
        },
        "ex": {
            "context_length": 20,
            "target_length": 40
        },
        "sea": {
            "context_length": 70,
            "target_length": 140
        },
        "full_sea": {
            "context_length": 10,
            "target_length": 20
        }
    }
}