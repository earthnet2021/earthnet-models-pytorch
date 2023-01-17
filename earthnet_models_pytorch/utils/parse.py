"""Utilities for parsing setting yamls

parse setting implements copying of global attributes to the particular subcategory.

"""

from pathlib import Path
import yaml
import warnings
from earthnet_models_pytorch.datamodule import DATAMODULES, METRIC_CHECKPOINT_INFO
from earthnet_models_pytorch.model import MODELS

# SETTINGS = ["en21-std", "en21-veg", "europe-veg", "en21x","en21x-px", "en22", "en23"]


def parse_setting(setting_file, track=None):

    setting_file = Path(setting_file)

    # decompose the path of the setting_file (in configs/)
    try:
        if "grid" == setting_file.parts[-2]:
            setting, architecture, feature, _, config = setting_file.parts[-5:]
        else:
            setting, architecture, feature, config = setting_file.parts[
                -4:
            ]  # example: setting: en21x, architecture: local-rnn, feature: arch, config: base.yaml
        config = "config_" + config[:-5] if config != "base.yaml" else "full_train"
        if setting not in DATAMODULES:
            setting = None
        if architecture not in MODELS:
            architecture = None
    except:
        setting, architecture, feature, config = None, None, None, None

    # Create the setting dict from the yaml file
    with open(setting_file, "r") as fp:
        setting_dict = yaml.load(fp, Loader=yaml.FullLoader)

    # check if the information of the setting_file path correspond with the yaml file
    if "Setting" in setting_dict:
        if setting is not None:
            if setting_dict["Setting"] != setting:
                warnings.warn(
                    f"Ambivalent definition of setting, found {setting_dict['Setting']} in yaml but {setting} in path. Using {setting_dict['Setting']}."
                )
                setting = setting_dict["Setting"]
    else:
        if setting is None:
            raise ValueError("No Setting was given.")
        setting_dict["Setting"] = setting
    setting_dict["Task"]["setting"] = setting_dict["Setting"]

    if "Architecture" in setting_dict:
        if architecture is not None:
            if setting_dict["Architecture"] != architecture:
                warnings.warn(
                    f"Ambivalent definition of model, found {setting_dict['Architecture']} in yaml but {architecture} in path. Using {setting_dict['Architecture']}."
                )
                architecture = setting_dict["Architecture"]
    else:
        if architecture is None:
            raise ValueError("No Model architecture was given.")
        setting_dict["Architecture"] = architecture

    if "Logger" in setting_dict:
        if "save_dir" in setting_dict["Logger"]:
            save_dir = setting_dict["Logger"]["save_dir"]
            if (len(list(Path(save_dir).parts)) < 2) or (
                (Path(save_dir).parts[-2] != setting_dict["Setting"])
                and (Path(save_dir).parts[-1] != setting_dict["Architecture"])
            ):
                setting_dict["Logger"]["save_dir"] = str(
                    Path(save_dir)
                    / setting_dict["Setting"]
                    / setting_dict["Architecture"]
                )
        else:
            setting_dict["Logger"]["save_dir"] = str(
                Path("experiments/")
                / setting_dict["Setting"]
                / setting_dict["Architecture"]
            )
        if "name" in setting_dict["Logger"]:
            if setting_dict["Logger"]["name"] != feature:
                warnings.warn(
                    f"Ambivalent definition of logger experiment, found {setting_dict['Logger']['name']} in yaml but {feature} is the set feature. Using {setting_dict['Logger']['name']}."
                )
        elif feature is not None:
            setting_dict["Logger"]["name"] = feature
        if "version" in setting_dict["Logger"]:
            if config is not None:
                if setting_dict["Logger"]["version"] != config:
                    warnings.warn(
                        f"Ambivalent definition of logger experiment config, found {setting_dict['Logger']['version']} in yaml but {config} in the path. Using {setting_dict['Logger']['version']}."
                    )
        elif config is not None:
            setting_dict["Logger"]["version"] = config

    else:
        setting_dict["Logger"] = {
            "save_dir": str(Path("experiments/") / setting)
            if setting is not None
            else "experiments/",
            "name": setting_dict["Architecture"],
        }
        if feature is not None:
            setting_dict["Logger"]["version"] = feature

    # Additionnal information for the training, save as metadata in /experiences
    setting_dict["Seed"] = setting_dict["Seed"] if "Seed" in setting_dict else 42
    setting_dict["Data"]["val_split_seed"] = setting_dict["Seed"]

    setting_dict["Trainer"]["precision"] = (
        setting_dict["Trainer"]["precision"]
        if "precision" in setting_dict["Trainer"]
        else 32
    )  # binary floating-point computer number format
    setting_dict["Data"]["fp16"] = (
        setting_dict["Trainer"]["precision"] == 16
    )  # binary floating-point computer number format

    setting_dict["Checkpointer"] = (
        {
            **setting_dict["Checkpointer"],
            **METRIC_CHECKPOINT_INFO[setting_dict["Setting"]],
        }
        if "Checkpointer" in setting_dict
        else METRIC_CHECKPOINT_INFO[setting_dict["Setting"]]
    )

    bs = setting_dict["Data"]["train_batch_size"]
    devices = setting_dict["Trainer"]["devices"]
    ddp = setting_dict["Trainer"]["strategy"] == "ddp"

    optimizers = setting_dict["Task"]["optimization"]["optimizer"]
    for optimizer in optimizers:
        if "lr_per_sample" in optimizer:
            lr_per_sample = optimizer["lr_per_sample"]
            if isinstance(devices, list):
                lr = bs * (len(devices) * ddp + (1 - ddp)) * lr_per_sample
            else:
                lr = bs * (devices * ddp + (1 - ddp)) * lr_per_sample
            print("learning rate", lr)
            optimizer["args"]["lr"] = lr

    if track is not None:
        setting_dict["Data"]["test_track"] = track

        if "pred_dir" not in setting_dict["Task"]:
            setting_dict["Task"]["pred_dir"] = (
                Path(setting_dict["Logger"]["save_dir"])
                / setting_dict["Logger"]["name"]
                / setting_dict["Logger"]["version"]
                / "preds"
                / track
            )

    if setting_dict["Architecture"] in MODELS:
        setting_dict["Model"]["setting"] = setting_dict["Setting"]

    setting_dict["Model"]["context_length"] = setting_dict["Task"]["context_length"]
    setting_dict["Model"]["target_length"] = setting_dict["Task"]["target_length"]

    setting_dict["Task"]["train_batch_size"] = setting_dict["Data"]["train_batch_size"]
    setting_dict["Task"]["val_batch_size"] = setting_dict["Data"]["val_batch_size"]
    setting_dict["Task"]["test_batch_size"] = setting_dict["Data"]["test_batch_size"]

    if ("min_lc" and "max_lc") not in setting_dict["Task"]["loss"]:
        raise Warning(
            "min_lc and max_lc are not defined in the yaml file. Default values are min_lc=82 and max_lc=104."
        )

    return setting_dict
