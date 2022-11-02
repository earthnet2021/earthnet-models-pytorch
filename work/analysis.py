from argparse import ArgumentParser

from earthnet_models_pytorch.utils import parse_setting
from test import test_model
import glob
import os
import seaborn as sns
import numpy as np
import json
import sys
import pandas as pd
import matplotlib.pyplot as plt

def test(args):
    # Prediction for each model
    for model in args.models:
        path_setting = args.path_setting + args.architecture + model + '/full_train/setting.yaml'
        print(args.path_setting + args.architecture + model + '/full_train/checkpoints/Epoch-epoch=*')
        path_checkpoint = glob.glob(args.path_setting + args.architecture + model + '/full_train/checkpoints/Epoch-epoch=*').pop()
        pred_dir = args.pred_dir + args.architecture + '/' + model
        setting_dict = parse_setting(path_setting, track = args.track)
        #if not os.path.exists(pred_dir):
        #        os.makedirs(pred_dir)

        if args.pred_dir is not None:
            setting_dict["Task"]["pred_dir"] = pred_dir

        test_model(setting_dict, path_checkpoint)   

    return
 


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--architecture', nargs = '+', type = str, default='dumby_mlp/', help='architecture of the model')
    parser.add_argument('--path_setting', type = str, default='experiments/en22/', help='path of yaml setting file')
    parser.add_argument('--track', type = str, default='iid', metavar='iid|ood|ex|sea', help='which track to test: either iid, ood, ex or sea')
    parser.add_argument('models', nargs='+', type = str, default=[], metavar='path/to/setting.yaml', help='list of models for the analysis')
    parser.add_argument('--pred_dir', type = str, default = '/workspace/data/UC1/L2_minicubes/prediction/en22/', metavar = 'path/to/predictions/directory/', help = 'Path where to save predictions') # prediction/en22/context-convlstm/ablation
    
    args = parser.parse_args()

    # Disabling PyTorch Lightning automatic SLURM detection
    for k, v in os.environ.items():
        if k.startswith("SLURM"):
            del os.environ[k]

    #violon_plot(args)
    test(args)