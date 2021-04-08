#!/usr/bin/env python3
"""Test Script
"""

import yaml

from argparse import ArgumentParser

import torch
import pytorch_lightning as pl

import json

from earthnet_models_pytorch.model import MODELS, MODELTASKS, MODELTASKNAMES
from earthnet_models_pytorch.setting import DATASETS
from earthnet_models_pytorch.utils import parse_setting
from earthnet_models_pytorch.task import TRACK_INFO

def test_model(setting_dict: dict, checkpoint: str):


    # Data
    data_args = ["--{}={}".format(key,value) for key, value in setting_dict["Data"].items()]
    data_parser = ArgumentParser()
    data_parser = DATASETS[setting_dict["Setting"]].add_data_specific_args(data_parser)
    data_params = data_parser.parse_args(data_args)
    dm = DATASETS[setting_dict["Setting"]](data_params)

    # Model
    model_args = ["--{}={}".format(key,value) for key, value in setting_dict["Model"].items()]
    model_parser = ArgumentParser()
    model_parser = MODELS[setting_dict["Architecture"]].add_model_specific_args(model_parser)
    model_params = model_parser.parse_args(model_args)
    model = MODELS[setting_dict["Architecture"]](model_params)

    # Task
    task_args = ["--{}={}".format(key,value) for key, value in setting_dict["Task"].items()]
    task_parser = ArgumentParser()
    task_parser = MODELTASKS[setting_dict["Architecture"]].add_task_specific_args(task_parser)
    task_params = task_parser.parse_args(task_args)
    task = MODELTASKS[setting_dict["Architecture"]](model = model, hparams = task_params)
    if checkpoint != "None":
        task.load_from_checkpoint(checkpoint_path = checkpoint, context_length = setting_dict["Task"]["context_length"], target_length = setting_dict["Task"]["target_length"], model = model, hparams = task_params)

    # Trainer
    trainer_dict = setting_dict["Trainer"]
    trainer_dict["logger"] = False
    trainer = pl.Trainer(**trainer_dict)

    dm.setup("test")
    trainer.test(model = task, datamodule = dm, ckpt_path = None)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('setting', type = str, metavar='path/to/setting.yaml', help='yaml with all settings')
    parser.add_argument('checkpoint', type = str, metavar='path/to/checkpoint', help='checkpoint file')
    parser.add_argument('track', type = str, metavar='iid|ood|ex|sea', help='which track to test: either iid, ood, ex or sea')
    parser.add_argument('--pred_dir', type = str, default = None, metavar = 'path/to/predictions/directory/', help = 'Path where to save predictions')
    args = parser.parse_args()

    import os
    for k, v in os.environ.items():
        if k.startswith("SLURM"):
            del os.environ[k]

    setting_dict = parse_setting(args.setting, track = args.track)

    if args.pred_dir is not None:
        setting_dict["Task"]["pred_dir"] = args.pred_dir

    test_model(setting_dict, args.checkpoint)
    