"""Utilities for parsing setting yamls

parse setting implements copying of global attributes to the particular subcategory.

"""

from pathlib import Path
import yaml
import warnings
from earthnet_models_pytorch.data import SETTINGS, METRIC_CHECKPOINT_INFO
from earthnet_models_pytorch.model import MODELS


def parse_setting(setting_file, track = None):
  

    setting_file = Path(setting_file)
    
    # decompose the path of the setting_file (in configs/)
    try:
        if "grid" == setting_file.parts[-2]:
            setting, architecture, feature, _, config = setting_file.parts[-5:]  
        else:
            setting, architecture, feature, config = setting_file.parts[-4:]  # example: setting: en21x, architecture: local-rnn, feature: arch, config: base.yaml
        config = "config_"+config[:-5] if config != "base.yaml" else "full_train"
        if setting not in SETTINGS: 
            setting = None
        if architecture not in MODELS:
            architecture = None
    except:
        setting, architecture, feature, config = None, None, None, None

    with open(setting_file, 'r') as fp:
        setting_dict = yaml.load(fp, Loader = yaml.FullLoader)

    # check if the information of the setting_file path correspond with the yaml file
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

    setting_dict["Trainer"]["precision"] = setting_dict["Trainer"]["precision"] if "precision" in setting_dict["Trainer"] else 32  # binary floating-point computer number format 
    setting_dict["Data"]["fp16"] = (setting_dict["Trainer"]["precision"] == 16)   # binary floating-point computer number format 
    setting_dict["Checkpointer"] = {**setting_dict["Checkpointer"], **METRIC_CHECKPOINT_INFO[setting_dict["Setting"]]} if "Checkpointer" in setting_dict else METRIC_CHECKPOINT_INFO[setting_dict["Setting"]]
    
    bs = setting_dict["Data"]["train_batch_size"]
    gpus = setting_dict["Trainer"]["gpus"] if "gpus" in setting_dict["Trainer"] else setting_dict["Trainer"]["devices"]
    ddp = (setting_dict["Trainer"]["strategy"] == "ddp")
    
    optimizers = setting_dict["Task"]["optimization"]["optimizer"]
    for optimizer in optimizers:  
        if "lr_per_sample" in optimizer:
            lr_per_sample = optimizer["lr_per_sample"]
            if isinstance(gpus, list):
                lr = bs * (len(gpus) * ddp + (1-ddp)) * lr_per_sample
            else:    
                lr = bs * (gpus * ddp + (1-ddp)) * lr_per_sample
            print('learning rate', lr)
            optimizer["args"]["lr"] = lr
        
    if track is not None: 
        setting_dict["Data"]["test_track"] = track

        if "pred_dir" not in setting_dict["Task"]:
            setting_dict["Task"]["pred_dir"] = Path(setting_dict["Logger"]["save_dir"])/setting_dict["Logger"]["name"]/setting_dict["Logger"]["version"]/"preds"/track

    if setting_dict["Architecture"] in ["channel-u-net", "local-rnn","rnn", "context-convlstm", "u-net-convlstm", "dumby-mlp", "convlstm-lstm"]:  
        setting_dict["Model"]["setting"] = setting_dict["Setting"]

    # Cpy information for others modules   
    setting_dict["Model"]["context_length"] = setting_dict["Task"]["context_length"]        
    setting_dict["Model"]["target_length"] = setting_dict["Task"]["target_length"]
    if "target" in setting_dict["Data"]:
        setting_dict["Model"]["target"] = setting_dict["Data"]["target"]

    setting_dict["Task"]["loss"]["context_length"] = setting_dict["Task"]["context_length"]        
    setting_dict["Task"]["loss"]["target_length"] = setting_dict["Task"]["target_length"]
        
    setting_dict["Task"]["train_batch_size"] = setting_dict["Data"]["train_batch_size"]
    setting_dict["Task"]["val_batch_size"] = setting_dict["Data"]["val_batch_size"]
    setting_dict["Task"]["test_batch_size"] = setting_dict["Data"]["test_batch_size"]

    setting_dict["Task"]["lc_min"] = setting_dict["Task"]["loss"]["lc_min"]
    setting_dict["Task"]["lc_max"] = setting_dict["Task"]["loss"]["lc_max"]

    setting_dict["Task"]["loss"]["setting"] = setting_dict["Setting"]
    
    if "metric_kwargs" not in setting_dict["Task"]:
        setting_dict["Task"]["metric_kwargs"] = {}
    setting_dict["Task"]["metric_kwargs"]["context_length"] = setting_dict["Task"]["context_length"]        
    setting_dict["Task"]["metric_kwargs"]["target_length"] = setting_dict["Task"]["target_length"]        
    setting_dict["Task"]["metric_kwargs"]["lc_min"] = setting_dict["Task"]["lc_min"] 
    setting_dict["Task"]["metric_kwargs"]["lc_max"] = setting_dict["Task"]["lc_max"] 

    if "model_shedules" in setting_dict["Task"]["metric_kwargs"]:
        setting_dict["Task"]["metric_kwargs"]["shedulers"] = setting_dict["Task"]["metric_kwargs"]["model_shedules"]

    return setting_dict