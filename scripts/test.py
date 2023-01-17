#!/usr/bin/env python3
"""Test Script
"""

from argparse import ArgumentParser

import pytorch_lightning as pl
import torch

from earthnet_models_pytorch.model import MODELS
from earthnet_models_pytorch.task.spatio_temporal import SpatioTemporalTask
from earthnet_models_pytorch.datamodule import DATAMODULES
from earthnet_models_pytorch.utils import parse_setting

# from torchsummary import summary
# from earthnet_models_pytorch.task import TRACK_INFO


def test_model(setting_dict: dict, checkpoint: str):

    # Data
    data_args = [
        "--{}={}".format(key, value) for key, value in setting_dict["Data"].items()
    ]
    data_parser = ArgumentParser()
    data_parser = DATAMODULES[setting_dict["Setting"]].add_data_specific_args(data_parser)
    data_params = data_parser.parse_args(data_args)
    dm = DATAMODULES[setting_dict["Setting"]](data_params)

    # Model
    model_args = [
        "--{}={}".format(key, value) for key, value in setting_dict["Model"].items()
    ]
    model_parser = ArgumentParser()
    model_parser = MODELS[setting_dict["Architecture"]].add_model_specific_args(
        model_parser
    )
    model_params = model_parser.parse_args(model_args)
    model = MODELS[setting_dict["Architecture"]](model_params)

    # Task
    task_args = [
        "--{}={}".format(key, value) for key, value in setting_dict["Task"].items()
    ]
    task_parser = ArgumentParser()
    task_parser = SpatioTemporalTask.add_task_specific_args(task_parser)
    task_params = task_parser.parse_args(task_args)
    task = SpatioTemporalTask(
        model=model, hparams=task_params
    )

    if checkpoint != "None":
        task.load_from_checkpoint(
            checkpoint_path=checkpoint,
            context_length=setting_dict["Task"]["context_length"],
            target_length=setting_dict["Task"]["target_length"],
            model=model,
            hparams=task_params,
        )

    # Trainer
    trainer_dict = setting_dict["Trainer"]
    trainer_dict["logger"] = False
    trainer = pl.Trainer(**trainer_dict)

    

    #x = next(iter(dm.setup("test"))) #torch.randn(4, 36, 26, 128, 128, device="cuda")
    #
    #torch.onnx.export(
    #    model,  # model being run
    #    x,  # model input (or a tuple for multiple inputs)
    #    str(setting_dict["Task"]["pred_dir"])
    #    + "/"
    #    + str(setting_dict["Logger"]["name"])
    #    + ".onnx",  # where to save the model (can be a file or file-like object)
    #    export_params=True,  # store the trained parameter weights inside the model file
    #    opset_version=10,  # the ONNX version to export the model to
    #    do_constant_folding=True,  # whether to execute constant folding for optimization
    #    input_names=["input"],  # the model's input names
    #    output_names=["output"],  # the model's output names
    #    dynamic_axes={
    #        "input": {
    #            0: "batch_size",
    #            1: "time",
    #            2: "channels",
    #            3: "lon",
    #            4: "lat",
    #        },  # variable length axes
    #        "output": {0: "batch_size"},
    #    },
    #)

    trainer.test(model=task, datamodule=dm, ckpt_path=None)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "setting",
        type=str,
        metavar="path/to/setting.yaml",
        help="yaml with all settings",
    )
    parser.add_argument(
        "checkpoint", type=str, metavar="path/to/checkpoint", help="checkpoint file"
    )
    parser.add_argument(
        "track",
        type=str,
        metavar="iid|ood|ex|sea",
        help="which track to test: either iid, ood, ex or sea",
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        default=None,
        metavar="/workspace/data/UC1/L2_minicubes/prediction/en22/",
        help="Path where to save predictions",
    )
    args = parser.parse_args()

    import os

    for k, v in os.environ.items():
        if k.startswith("SLURM"):
            del os.environ[k]

    setting_dict = parse_setting(args.setting, track=args.track)

    if args.pred_dir is not None:
        setting_dict["Task"]["pred_dir"] = args.pred_dir

    test_model(setting_dict, args.checkpoint)
