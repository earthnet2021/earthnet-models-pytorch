from argparse import ArgumentParser
import matplotlib.colors as clr

import torch
import os
from pathlib import Path
from earthnet_models_pytorch.model import MODELS
from earthnet_models_pytorch.task.spatio_temporal import SpatioTemporalTask
from earthnet_models_pytorch.datamodule import DATASETS
from earthnet_models_pytorch.metric import NormalizedNashSutcliffeEfficiency
from earthnet_models_pytorch.utils import parse_setting


import streamlit as st

st.set_page_config(layout="wide")


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

def visualisation_targ(targs):
    sentinel = targs["dynamic"][0]
    ndvi = sentinel[0, -10:, 0, ...]
    rgb = torch.cat(
        [
            sentinel[0, -10:, 3, ...].unsqueeze(3) * 10,
            sentinel[0, -10:, 2, ...].unsqueeze(3) * 10,
            sentinel[0, -10:, 1, ...].unsqueeze(3) * 10,
        ],
        dim=3,
    ).cpu().detach().numpy()
    print(rgb)
    grid = ndvi.squeeze(2).squeeze(0).cpu().detach().numpy()
    grid = cmap_veg(grid)
    
    st.write("Satellite data")
    cols = st.columns(grid.shape[0])
    for col_num in range(grid.shape[0]):
        cols[col_num].image(rgb[col_num, ...], clamp=True, channels='RGB', use_column_width=True)

    st.write("Target")
    cols = st.columns(grid.shape[0])
    for col_num in range(grid.shape[0]):
        cols[col_num].image(grid[col_num, ...], use_column_width=True)

    return

def visualisation_pred(preds, score):
    grid = preds.squeeze(2).squeeze(0).cpu().detach().numpy()
    grid = cmap_veg(grid)

    cols = st.columns(grid.shape[0])
    for col_num in range(grid.shape[0]):
        cols[col_num].image(grid[col_num, ...], use_column_width=True)
    return


if __name__ == "__main__":
    setting = os.getcwd() + "/demo/setting.yaml"
    checkpoint = os.getcwd() + "/demo/last.ckpt"
    track = "iid"

    setting_dict = parse_setting(setting, track)

    st.title("Forecast vegetation greeness on the Earthnet2023 dataset")
    minicube = load_minicube()
    model = load_model(checkpoint)

    result = st.button("Run on minicube")
    if result:    
        visualisation_targ(minicube)
        with st.empty():
            st.write("Calculating prediction...")
            with torch.no_grad():
                prediction, _ = model(minicube, 1)
            # score = metric(prediction, minicube)

            score = 1
            st.write("Prediction")
        visualisation_pred(prediction, score)
