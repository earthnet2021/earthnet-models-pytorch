from argparse import ArgumentParser

import pytorch_lightning as pl
import torch

from earthnet_models_pytorch.model import MODELS
from earthnet_models_pytorch.task.spatio_temporal import SpatioTemporalTask
from earthnet_models_pytorch.datamodule import DATASETS
from earthnet_models_pytorch.utils import parse_setting


def predict_model(setting_dict: dict, checkpoint: str):

    # Data
    data_args = [
        "--{}={}".format(key, value) for key, value in setting_dict["Data"].items()
    ]
    data_parser = ArgumentParser()
    data_parser = DATASETS[setting_dict["Setting"]].add_data_specific_args(data_parser)
    data_params = data_parser.parse_args(data_args)
    dm = DATASETS[setting_dict["Setting"]](data_params)

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
    task = SpatioTemporalTask(model=model, hparams=task_params)

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

    dm.setup("test")

    x = next(
        iter(dm.test_dataloader())
    )  # torch.randn(4, 36, 26, 128, 128, device="cuda")

    trainer.predict(model=task, datamodule=dm, ckpt_path=None)


if __name__ == "__main__":
    setting = "experiments/en22/context-convlstm/baseline_RGBNR/full_train/setting.yaml"
    checkpoint = "experiments/en22/context-convlstm/baseline_RGBNR/full_train/checkpoints/last.ckpt"
    track = "iid"

    setting_dict = parse_setting(setting, track)

    predict_model(setting_dict, checkpoint)
