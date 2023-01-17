from argparse import ArgumentParser
import matplotlib.colors as clr

import pytorch_lightning as pl
import torch
import os
from pathlib import Path
import torchvision
from PIL import Image
from earthnet_models_pytorch.model import MODELS
from earthnet_models_pytorch.task.spatio_temporal import SpatioTemporalTask
from earthnet_models_pytorch.datamodule import DATASETS
from earthnet_models_pytorch.metric import NormalizedNashSutcliffeEfficiency
from earthnet_models_pytorch.utils import parse_setting
from torch.utils.data import Dataset, DataLoader


import streamlit as st


cmap_veg = clr.LinearSegmentedColormap.from_list(
    "veg",
    [
        "#ffffe5",
        "#f7fcb9",
        "#d9f0a3",
        "#addd8e",
        "#78c679",
        "#41ab5d",
        "#238443",
        "#006837",
        "#004529",
    ],
)


def load_minicube():
    base_dir = Path(os.getcwd() + "/demo/test_dataset/")
    dataset = DATASETS[setting_dict["Setting"]](base_dir)

    # Create a dict name: index
    filepaths = sorted(list(base_dir.glob("*.nc")))
    dict_index = dict()
    for index, filepath in enumerate(filepaths):
        dict_index[filepath.name] = index

    # Select widget
    minicube_name = st.selectbox("Select a minicube", tuple(dict_index.keys()))
    data = dataset[dict_index[minicube_name]]

    data["dynamic"][0] = data["dynamic"][0].unsqueeze(0)
    data["dynamic"][1] = data["dynamic"][1].unsqueeze(0)
    data["static"][0] = data["static"][0].unsqueeze(0)
    data["landcover"] = data["landcover"].unsqueeze(0)
    return data


def load_model(checkpoint):
    model_args = [
        "--{}={}".format(key, value) for key, value in setting_dict["Model"].items()
    ]
    model_parser = ArgumentParser()
    model_parser = MODELS[setting_dict["Architecture"]].add_model_specific_args(
        model_parser
    )
    model_params = model_parser.parse_args(model_args)
    model = MODELS[setting_dict["Architecture"]](model_params)
    # checkpoint_dict = torch.load(checkpoint)

    # Task
    task_args = [
        "--{}={}".format(key, value) for key, value in setting_dict["Task"].items()
    ]
    task_parser = ArgumentParser()
    task_parser = SpatioTemporalTask.add_task_specific_args(task_parser)
    task_params = task_parser.parse_args(task_args)
    task = SpatioTemporalTask(model=model, hparams=task_params)

    task.load_from_checkpoint(
        checkpoint_path=checkpoint,
        context_length=setting_dict["Task"]["context_length"],
        target_length=setting_dict["Task"]["target_length"],
        model=model,
        hparams=task_params,
    )

    task.eval()
    return task


def metric(prediction, minicube):
    metric = NormalizedNashSutcliffeEfficiency(min_lc=2, max_lc=6, batch_size=1)
    score = metric.compute_sample(prediction, minicube)
    return score


def visualisation(targs, preds, score):
    sentinel = targs["dynamic"][0]
    lc = targs["landcover"]
    nrow = 9 if sentinel.shape[1] % 9 == 0 else 10
    """rgb = torch.cat(
        [
            sentinel[0, -preds.shape[1], 2, ...].unsqueeze(1) * 10000,
            sentinel[0, -preds.shape[1], 1, ...].unsqueeze(1) * 10000,
            sentinel[0, -preds.shape[1], 0, ...].unsqueeze(1) * 10000,
        ],
        dim=1,
    )"""
    grid = preds.squeeze(2).squeeze(0).cpu().detach().numpy()
    # grid = cmap_veg(grid)
    st.image(grid[1, ...])
    # cols = st.columns(grid.shape[0])
    # for col_num in range(grid.shape[0]):
    # img = Image.fromarray(grid[col_num, ...])
    # print(grid[col_num, ...].shape)
    # cols[col_num].image(grid[col_num, ...])

    return


if __name__ == "__main__":
    setting = "experiments/en23/context-convlstm/baseline_RGBNR/full_train/setting.yaml"
    checkpoint = "experiments/en23/context-convlstm/baseline_RGBNR/full_train/checkpoints/last.ckpt"
    track = "iid"

    setting_dict = parse_setting(setting, track)

    st.title("Demo of a model for vegetation greeness on the Earthnet dataset")
    minicube = load_minicube()
    model = load_model(checkpoint)

    result = 1  # st.button("Run on minicube")
    if result:
        st.write("Calculating prediction...")
        with torch.no_grad():
            prediction, _ = model(minicube, 1)
        # score = metric(prediction, minicube)
        score = 1
        visualisation(minicube, prediction, score)
