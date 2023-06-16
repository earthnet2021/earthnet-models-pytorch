#!/usr/bin/env python3
"""Debug script
"""
#TODO Add more output to textfile

from argparse import ArgumentParser
import shutil
import os
import time
import yaml
from pathlib import Path
import copy
import sys 
import torch
import pytorch_lightning as pl

from earthnet_models_pytorch.model import MODELS, MODELTASKS
from earthnet_models_pytorch.datamodule import DATASETS
from earthnet_models_pytorch.utils import parse_setting


def fast_dev_run(setting_dict: dict):

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
    task_params = task_parser.parse_args(task_args)   # argsparse.Namespace with the setting_dict["Task"] dictionnary
    task = MODELTASKS[setting_dict["Architecture"]](model = model, hparams = task_params) # call the SpacioTemporalTask module with the model module

    # Trainer
    trainer_dict = setting_dict["Trainer"]
    trainer_dict["logger"] = False
    trainer_dict["fast_dev_run"] = 2  # number of batch
    #trainer_dict["log_gpu_memory"] = 'all'
    if "profiler" in trainer_dict:
        trainer_dict["profiler"] = pl.profiler.AdvancedProfiler(output_filename="curr_profile") #TODO better output filename...  # performance analysis

    trainer = pl.Trainer(**trainer_dict)  # Customize every aspect of training via flags

    # dm.setup("fit") LightningDeprecationWarning: DataModule.setup has already been called, so it will not be called again. In v1.6 this behavior will change to always call DataModule.setup.
    trainer.fit(task, dm)  # (model, train_dataloader, val_dataloader)

    end = time.time()

    print(f"Calculation done in {end - start} seconds.")

    return "Fast dev run succeeded!\n"



def overfit_model(setting_dict: dict): 
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
    setting_dict["Task"]["optimization"]["lr_shedule"][0]["args"]["milestones"] = [1000,2000]
    task_args = ["--{}={}".format(key,value) for key, value in setting_dict["Task"].items()]
    task_parser = ArgumentParser()
    task_parser = MODELTASKS[setting_dict["Architecture"]].add_task_specific_args(task_parser)
    task_params = task_parser.parse_args(task_args)
    task = MODELTASKS[setting_dict["Architecture"]](model = model, hparams = task_params)
    
    # Logger
    setting_dict["Logger"]["version"] = "debug"
    logger = pl.loggers.TensorBoardLogger(**setting_dict["Logger"])

    # Trainer
    trainer_dict = setting_dict["Trainer"]
    trainer_dict["overfit_batches"] = 2  # Uses this much data of the training set.
    trainer_dict["check_val_every_n_epoch"] = 50  #Check val every n train epochs.
    trainer_dict["max_epochs"] = 10
    #trainer_dict["num_sanity_val_steps"] = 0  #Sanity check runs n batches of val before starting the training routine
    if "profiler" in trainer_dict:
        trainer_dict["profiler"] = pl.profiler.AdvancedProfiler(output_filename="curr_profile")

    trainer = pl.Trainer(logger = logger, **trainer_dict)

    dm.setup("fit")
    trainer.fit(task, dm)  # (model, train_dataloader, val_dataloader)


    end = time.time()

    print(f"Calculation done in {end - start} seconds.")

    return "Overfitting done!\n"



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('setting', type = str, metavar='path/to/setting.yaml', help='yaml with all settings')
    args = parser.parse_args()

    # Disabling PyTorch Lightning automatic SLURM detection
    for k, v in os.environ.items():  # set of environment variables
        if k.startswith("SLURM"):
            del os.environ[k]

    setting_dict = parse_setting(args.setting)  # run parse_setting on the path of the setting, return 
    outtext = "Starting fast dev run....\n"

    outtext += fast_dev_run(copy.deepcopy(setting_dict))
    torch.cuda.empty_cache()

    outtext += "Starting overfitting....\n"

    outtext += overfit_model(copy.deepcopy(setting_dict))

    outpath = Path(setting_dict["Logger"]["save_dir"])/setting_dict["Logger"]["name"]/"debug/debug_results.txt"

    with open(outpath, 'w') as f:
        f.write(outtext)
