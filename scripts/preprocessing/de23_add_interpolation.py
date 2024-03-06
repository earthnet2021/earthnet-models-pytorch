import zarr
import pandas as pd
from pathlib import Path
import datetime
import xarray as xr
import numpy as np
from tqdm import tqdm
import random
import multiprocessing as mp
from scipy.interpolate import interp1d
import os
import sys

spatio_temporal_variables = ["B02", "B03", "B04", "B05", "B06", "B07", "B8A"]


def get_dataset():
    test_fold = 10
    val_fold = 9

    # load csv
    df = pd.read_csv(
        "/Net/Groups/BGI/work_2/scratch/DeepExtremes/mc_earthnet.csv", delimiter=","
    )[["path", "group", "check", "start_date"]]

    # 3 training minicubes between 2017 - 2020 (randomly starting in the first year), followed by 3 test minicubes in 2021
    df["start_date"] = pd.to_datetime(df["start_date"], format="%Y-%m-%dT%H:%M:%S.%f")
    df["start_date2"] = df["start_date"] + datetime.timedelta(days=450)
    df["start_date3"] = df["start_date2"] + datetime.timedelta(days=450)
    df["start_test1"] = df["start_date3"] + datetime.timedelta(days=450)
    df["start_test2"] = df["start_test1"] + datetime.timedelta(days=90)
    df["start_test3"] = df["start_test2"] + datetime.timedelta(days=90)
    df = df.melt(
        ["path", "group", "check"], value_name="start_date", var_name="variable"
    )
    # use only the context period
    df["end_date"] = df["start_date"] + datetime.timedelta(days=364)

    # temporal test set 2021
    temporal_test_subset = df.loc[
        (df["variable"].str.startswith("start_test")),
        ["path", "start_date", "end_date"],
    ]

    # folds 2017 - 2020
    df = df.loc[df["variable"].str.startswith("start_date")].drop(columns="variable")

    # training set
    train_subset = df.loc[
        (df["group"] != test_fold) & (df["group"] != val_fold) & (df["check"] == 0),
        ["path", "start_date", "end_date"],
    ]

    # validation set
    val_subset = df.loc[
        (df["group"] == val_fold) & (df["check"] == 0),
        ["path", "start_date", "end_date"],
    ]

    # iid test set
    spatial_test_subset = df.loc[
        (df["group"] == test_fold) & (df["check"] == 0),
        ["path", "start_date", "end_date"],
    ]

    return train_subset, val_subset, spatial_test_subset, temporal_test_subset


def interpolate_temporal(data):
    # Get the indices along the time_steps dimension
    time_indices = np.arange(len(data))
    # Create a mask for non-missing values
    mask = ~np.isnan(data)

    if mask.any():
        # If the first frame has missing values, fill them with the mean of the frame
        if ~mask[0]:
            data[0] = np.nanmean(data)
            mask[0] = True
        # If the last frame has missing values, fill them with the mean of the frame
        if ~mask[-1]:
            data[-1] = np.nanmean(data)
            mask[-1] = True

        # Use linear interpolation to fill missing values
        interp_func = interp1d(time_indices[mask], data[mask], kind="linear")
        data[~mask] = interp_func(time_indices[~mask])
    return data


def process_sample(metadata):
    filepath, start_date, end_date = metadata
    output_filepath = os.path.join(dst_path, str(filepath)[len(str(basepath)) + 1 :])
    if os.path.exists(output_filepath):
        filepath = output_filepath

    minicube_original = xr.open_dataset(filepath, engine="zarr")
    time = slice(start_date, end_date)
    event_time = slice(
        start_date, min(end_date, pd.Timestamp(datetime.date(2021, 12, 31)))
    )
    minicube = minicube_original.sel(
        time=time,
        event_time=event_time,
    )
    mask_cloud = minicube.cloudmask_en.values == 0
    mask_is_avail = minicube.SCL.values != 0

    for variable in spatio_temporal_variables:
        # Get the variable data from the minicube
        var_data = minicube[variable].values
        mask = mask_cloud * mask_is_avail
        # Apply temporal interpolation to fill missing values
        var_data = np.where(var_data * mask == 0, np.nan, var_data)
        # Apply interpolation to all i, j positions simultaneously
        if mask.any():
            interpolated_results = np.apply_along_axis(
                interpolate_temporal, 0, var_data
            )
            # Combine the interpolated data back into the minicube
            minicube_original[variable].loc[
                dict(
                    time=time,
                )
            ] = interpolated_results

    # Saving
    minicube_original.chunk().to_zarr(output_filepath, mode="w", consolidated=True)


if __name__ == "__main__":
    basepath = Path("/Net/Groups/BGI/tscratch/mweynants/dx-minicubes/")
    dst_path = "/Net/Groups/BGI/tscratch/crobin/dx-minicubes_interpolated/"

    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    folders = list(basepath.glob("full/*"))

    for folder in folders:
        path = os.path.join(dst_path, str(folder)[len(str(basepath)) + 1 :])
        if not os.path.exists(path):
            os.makedirs(path)

    (
        train_subset,
        val_subset,
        spatial_test_subset,
        temporal_test_subset,
    ) = get_dataset()

    metadata = []
    for metadata_files in [train_subset, val_subset, spatial_test_subset]:
        metadata += sorted(
            (
                Path(basepath, metadata_files["path"][idx][1:]),
                metadata_files["start_date"][idx],
                metadata_files["end_date"][idx],
            )
            for idx in metadata_files.index
        )

    with mp.Pool(1) as pool:
        r = list(tqdm(pool.imap(process_sample, metadata[:4]), total=len(metadata[:4])))
