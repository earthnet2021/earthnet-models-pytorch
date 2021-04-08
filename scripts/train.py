#!/usr/bin/env python3
"""Train Script
"""

from argparse import ArgumentParser
import shutil
import os
import time
import yaml

import torch
import pytorch_lightning as pl

from earthnet_models_pytorch.model import MODELS, MODELTASKS
from earthnet_models_pytorch.setting import DATASETS
from earthnet_models_pytorch.utils import parse_setting


def train_model(setting_dict: dict, setting_file: str = None):
    start = time.time()

    pl.seed_everything(setting_dict["Seed"])
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
    
    # Logger
    logger = pl.loggers.TensorBoardLogger(**setting_dict["Logger"])

    if setting_file is not None and type(logger.experiment).__name__ != "DummyExperiment":
        print("Copying setting yaml.")
        os.makedirs(logger.log_dir, exist_ok = True)
        with open(os.path.join(logger.log_dir,"setting.yaml"), 'w') as fp:
            yaml.dump(setting_dict, fp)

    # Checkpointing
    checkpoint_callback = pl.callbacks.ModelCheckpoint(**setting_dict["Checkpointer"])

    # Trainer
    trainer_dict = setting_dict["Trainer"]
    if "profiler" in trainer_dict:
        trainer_dict["profiler"] = pl.profiler.AdvancedProfiler(output_filename="curr_profile")

    trainer = pl.Trainer(logger = logger, callbacks = [checkpoint_callback], **trainer_dict)

    dm.setup("fit")
    trainer.fit(task, dm)
    print(f"Best model {checkpoint_callback.best_model_path} with score {checkpoint_callback.best_model_score}")

    end = time.time()

    print(f"Calculation done in {end - start} seconds.")
    




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('setting', type = str, metavar='path/to/setting.yaml', help='yaml with all settings')
    args = parser.parse_args()

    # Disabling PyTorch Lightning automatic SLURM detection
    for k, v in os.environ.items():
        if k.startswith("SLURM"):
            del os.environ[k]

    setting_dict = parse_setting(args.setting)

    train_model(setting_dict, args.setting)
    