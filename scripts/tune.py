#!/usr/bin/env python3
"""Tune script
"""


#TODO test in Dresden
#TODO 2 modes: overwrite existing / tune only non-existing

import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml
import pandas as pd
from earthnet_models_pytorch.utils import parse_setting


class Tuner:

    def __init__(self, slurm = False):
        self.slurm = slurm
        self.trial_paths = []

    @staticmethod
    def update_dict(in_dict, keys, val):
        if len(keys) > 0:
            out = Tuner.update_dict(in_dict[keys[0]], keys[1:], val)
            in_dict[keys[0]] = out
        else:
            in_dict = val
        return in_dict

    def create_tune_cfgs(self, params_file, setting_file):
        
        with open(setting_file, 'r') as fp:
            setting_dict = yaml.load(fp, Loader = yaml.FullLoader)

        with open(params_file, 'r') as fp:
            params_dict = yaml.load(fp, Loader = yaml.FullLoader)

        outpath = Path(params_file).parent/"grid"

        outpath.mkdir(parents = True, exist_ok = True)

        flat_params = [[{"param": param_name, "val": v} for v in param_vals] for param_name, param_vals in params_dict.items()]
        
        trials = list(itertools.product(*flat_params))

        trial_paths = []

        for trial in trials:
            trial_name = []
            for elem in trial:
                access = elem["param"].split(".")
                trial_name.append(f"({access[-1]}={elem['val']})")
                access = [int(key) if key.isdigit() else key for key in access]
                setting_dict = self.update_dict(setting_dict, access, elem['val'])

            trial_name = "_".join(trial_name)

            trial_path = outpath/(trial_name+".yaml")

            with open(trial_path, 'w') as fp:
                yaml.dump(setting_dict, fp)

            trial_paths.append(trial_path)

        self.trial_paths = trial_paths

        return trial_paths

    def start_one_trial(self, trial_path):
        # TODO export CUDA_VISIBLE_DEVICES ???
        
        script_path = Path(sys.path[1]).parent.parent/"bin"

        if not self.slurm:
            cmd = ["train.py", trial_path] #f"{script_path/'train.py'} {trial_path}"
        else:
            cmd = f"{script_path/'slurmrun.sh'} {trial_path}"
        out = subprocess.Popen(cmd)

        return out

    def run_trials(self):
        
        managers = []
        for trial_path in self.trial_paths:
            managers.append(self.start_one_trial(trial_path))
            if not self.slurm:
                managers[-1].wait()

        errors = ""

        done = [0]*len(managers)
        while not all(done):
            for idx, manager in enumerate(managers):
                manager.poll()
                if manager.returncode is not None:
                    if manager.returncode == 0:
                        done[idx] = True
                    else:
                        # managers[idx] = self.start_one_trial(self.trial_paths[idx]) # TODO this would be auto-restarting
                        done[idx] = True
                        errors += f"{self.trial_paths[idx]} failed with returncode {manager.returncode}"
        
        return errors
    
    def aggregate_results(self):

        all_scores = []
        best_scores = []

        for trial_path in self.trial_paths:
            setting_dict = parse_setting(trial_path)
            out_dir = Path(setting_dict["Logger"]["save_dir"])/setting_dict["Logger"]["name"]/setting_dict["Logger"]["version"]
            with open(out_dir/"validation_scores.json", 'r') as fp:
                scores = json.load(fp)
            if setting_dict["Setting"] == "en21-std":
                score = max([s["EarthNetScore"] for s in scores])
            else:
                score = min([s["RMSE_Veg"] for s in scores])

            best_scores.append(score)

            all_scores += [{**s, **{"trial": trial_path}} for s in scores]

        best_score = (max(best_scores) if setting_dict["Setting"] == "en21-std" else min(best_scores))
        best_trial = [self.trial_paths[i] for i, s in enumerate(best_scores) if (s == best_score)][0]

        cfg_path = Path(trial_path).parent.parent/"best.yaml"

        best_setting = parse_setting(best_trial)

        print(f"Best trial {best_trial} with score {best_score}.")

        with open(cfg_path, 'w') as fp:
            yaml.dump(best_setting, fp)
        
        scores_df = pd.DataFrame(all_scores)

        scores_df.to_csv(Path(best_setting["Logger"]["save_dir"])/best_setting["Logger"]["name"]/"tune_results.csv")
    

    @classmethod
    def tune(cls, params_file, setting_file, slurm = False):
        
        self = cls(slurm = slurm)

        print("Creating Configs...")
        self.create_tune_cfgs(params_file, setting_file)

        print("Running Trials...")
        #self.run_trials()

        print("Aggregating Results...")
        self.aggregate_results()


if __name__=="__main__":
    import fire
    fire.Fire(Tuner.tune)    

