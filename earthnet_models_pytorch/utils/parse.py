"""Utilities for parsing setting yamls

parse setting implements copying of global attributes to the particular subcategory.

"""

from pathlib import Path
import yaml
import warnings
from earthnet_models_pytorch.setting import SETTINGS, METRIC_CHECKPOINT_INFO
from earthnet_models_pytorch.model import MODELS, MODELTASKNAMES
from earthnet_models_pytorch.task import TRACK_INFO

def parse_setting(setting_file, track = None):

    setting_file = Path(setting_file)

    try:
        if "grid" == setting_file.parts[-2]:
            setting, architecture, feature, _, config = setting_file.parts[-5:]
        else:
            setting, architecture, feature, config = setting_file.parts[-4:]
        config = "config_"+config[:-5] if config != "base.yaml" else "full_train"
        if setting not in SETTINGS: 
            setting = None
        if architecture not in MODELS:
            architecture = None
    except:
        setting, architecture, feature, config = None, None, None, None

    with open(setting_file, 'r') as fp:
        setting_dict = yaml.load(fp, Loader = yaml.FullLoader)

    if "Setting" in setting_dict:
        if setting is not None:
            if setting_dict["Setting"] != setting:
                warnings.warn(f"Ambivalent definition of setting, found {setting_dict['Setting']} in yaml but {setting} in path. Using {setting_dict['Setting']}.")
                setting = setting_dict["Setting"]
    else:
        setting_dict["Setting"] = setting if setting is not None else "en21-std"
    setting_dict["Task"]["setting"] = setting_dict["Setting"]

    if "Architecture" in setting_dict:
        if architecture is not None:
            if setting_dict["Architecture"] != architecture:
                warnings.warn(f"Ambivalent definition of model, found {setting_dict['Architecture']} in yaml but {architecture} in path. Using {setting_dict['Architecture']}.")
                architecture = setting_dict["Architecture"]
    else:
        if architecture is None:
            raise ValueError('No Model architecture was given.')
        setting_dict["Architecture"] = architecture


    if "Logger" in setting_dict:
        if "save_dir" in setting_dict["Logger"]:
            save_dir = setting_dict["Logger"]["save_dir"]
            if (len(list(Path(save_dir).parts)) < 2) or ((Path(save_dir).parts[-2] != setting_dict["Setting"]) and (Path(save_dir).parts[-1] != setting_dict["Architecture"])):
                setting_dict["Logger"]["save_dir"] = str(Path(save_dir)/setting_dict["Setting"]/setting_dict["Architecture"])
        else:
            setting_dict["Logger"]["save_dir"] = str(Path("experiments/")/setting_dict["Setting"]/setting_dict["Architecture"])
        if "name" in setting_dict["Logger"]:
            if setting_dict["Logger"]["name"] != feature:
                warnings.warn(f"Ambivalent definition of logger experiment, found {setting_dict['Logger']['name']} in yaml but {feature} is the set feature. Using {setting_dict['Logger']['name']}.")
        elif feature is not None:
            setting_dict["Logger"]["name"] = feature
        if "version" in setting_dict["Logger"]:
            if config is not None:
                if setting_dict["Logger"]["version"] != config:
                    warnings.warn(f"Ambivalent definition of logger experiment config, found {setting_dict['Logger']['version']} in yaml but {config} in the path. Using {setting_dict['Logger']['version']}.")
        elif config is not None:
            setting_dict["Logger"]["version"] = config

    else:
        setting_dict["Logger"] = {
            "save_dir": str(Path("experiments/")/setting) if setting is not None else "experiments/",
            "name": setting_dict["Architecture"],
            }
        if feature is not None:
            setting_dict["Logger"]["version"] = feature
        

    setting_dict["Seed"] = setting_dict["Seed"] if "Seed" in setting_dict else 42
    setting_dict["Data"]["val_split_seed"] = setting_dict["Seed"]

    setting_dict["Trainer"]["precision"] = setting_dict["Trainer"]["precision"] if "precision" in setting_dict["Trainer"] else 32
    setting_dict["Data"]["fp16"] = (setting_dict["Trainer"]["precision"] == 16)

    setting_dict["Checkpointer"] = {**setting_dict["Checkpointer"], **METRIC_CHECKPOINT_INFO[setting_dict["Setting"]]} if "Checkpointer" in setting_dict else METRIC_CHECKPOINT_INFO[setting_dict["Setting"]]

    bs = setting_dict["Data"]["train_batch_size"]
    gpus = setting_dict["Trainer"]["gpus"]
    ddp = (setting_dict["Trainer"]["accelerator"] == "ddp")
    optimizers = setting_dict["Task"]["optimization"]["optimizer"]
    for optimizer in optimizers:
        if "lr_per_sample" in optimizer:
            lr_per_sample = optimizer["lr_per_sample"]
            lr = bs * (gpus * ddp + (1-ddp)) * lr_per_sample
            optimizer["args"]["lr"] = lr
        
    if track is not None:

        setting_dict["Task"]["context_length"] = TRACK_INFO[setting_dict["Setting"]][track]["context_length"]
        setting_dict["Task"]["target_length"] = TRACK_INFO[setting_dict["Setting"]][track]["target_length"]

        setting_dict["Data"]["test_track"] = track

        if "pred_dir" not in setting_dict["Task"]:
            setting_dict["Task"]["pred_dir"] = Path(setting_dict["Logger"]["save_dir"])/setting_dict["Logger"]["name"]/setting_dict["Logger"]["version"]/"preds"/track

    if setting_dict["Architecture"] in ["channel-u-net", "local-rnn","rnn"]:
        setting_dict["Model"]["setting"] = setting_dict["Setting"]

    return setting_dict