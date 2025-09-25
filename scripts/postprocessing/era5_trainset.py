from pathlib import Path
import xarray as xr
import numpy as np
import json
from tqdm import tqdm
import multiprocessing as mp
import os
from scipy.stats import pearsonr
from argparse import ArgumentParser
import sys
import pandas as pd

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
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

era5 = [
    "t2m_mean",
    "pev_mean",
    "slhf_mean",
    "ssr_mean",
    "sp_mean",
    "sshf_mean",
    "e_mean",
    "tp_mean",
    "t2m_min",
    "pev_min",
    "slhf_min",
    "ssr_min",
    "sp_min",
    "sshf_min",
    "e_min",
    "tp_min",
    "t2m_max",
    "pev_max",
    "slhf_max",
    "ssr_max",
    "sp_max",
    "sshf_max",
    "e_max",
    "tp_max",
]


def calculate_variable_statistics(
    target,
    cubename="",
    method="by_frame",
):
    """Function to calculate statistics for each variable in a sample"""
    # set mask = 1 where data are OK
    cloud_mask = target.cloudmask_en == 0
    lc = target.SCL
    lc_mask = lc == 4
    mask = cloud_mask * lc_mask
    targ = (target.B8A - target.B04) / (target.B8A + target.B04 + 1e-8)

    targ = np.where(targ * mask == 0, np.nan, targ)
    n_obs = np.count_nonzero(~np.isnan(targ), axis=0)
    if np.sum(n_obs) / (128 * 128 * 17) < 0.05:
        # not enough data
        return (
            cubename,
            {},
        )
    mean_targ = np.zeros((128, 128))
    sum_squared_dev = np.zeros((128, 128))
    for i in range(128):
        for j in range(128):
            x = targ[:, i, j]

            nas = np.isnan(x)
            if len(x[~nas]) > 1 and ~np.all(x[~nas] == x[~nas][0]):
                x = x[~nas]
                mean = np.mean(x)
                dev_targ = x - mean
                dev_targ_squared = dev_targ**2
                mean_targ[i, j] = mean
                sum_squared_dev[i, j] = np.sum(dev_targ_squared)
            else:
                sum_squared_dev[i, j] = np.nan
    mean_targ = (
        np.nanmean(mean_targ) if np.count_nonzero(~np.isnan(mean_targ)) != 0 else None
    )
    sum_squared_dev = (
        np.nanmean(sum_squared_dev)
        if np.count_nonzero(~np.isnan(sum_squared_dev)) != 0
        else None
    )

    # Meteorological variables
    era5_stats = {}
    for variable in era5:
        if variable[-3:] == "min":
            era5_stats[variable] = np.float64(np.nanmin(target[variable]))
        elif variable[-3:] == "max":
            era5_stats[variable] = np.float64(np.nanmax(target[variable]))
        else:
            era5_stats[variable] = np.float64(np.nanmean(target[variable]))

    # NDVI
    if method == "by_frame":
        # stats are first computed over time and then averaged over space by minicube
        result = {
            "mean_targ": np.float64(mean_targ),
            "sigma_squared_targ": np.float64(sum_squared_dev),
        }
        result.update(era5_stats)
        return (
            cubename,
            result,
        )


def calculate_sample_statistics(data, method="by_frame"):
    cubename_target = data["path"]
    end_date = data["end_date"]
    start_date = end_date - datetime.timedelta(days=17 * 5)

    # try:
    # get test paths
    test_path = Path(
        "/Net/Groups/BGI/tscratch/mweynants/dx-minicubes", cubename_target[1:]
    )
    target = (
        xr.open_dataset(test_path, engine="zarr")
        .sel(time=slice(start_date, end_date))
        .load()
    )
    return calculate_variable_statistics(target, cubename_target, method)


def calculate(args):
    result = calculate_sample_statistics(*args)
    return result


def process_samples_in_parallel(paths, method="by_frame"):
    "Function to process samples using parallel processing"
    num_cores = mp.cpu_count()
    print("numbers of cores availables: ", num_cores)
    pool = mp.Pool(processes=100)
    tasks = [(paths.iloc[i], method) for i in range(paths.shape[0])]
    results = list(
        # tqdm(pool.imap(calculate_sample_statistics, paths,), total=len(paths))
        tqdm(pool.imap(calculate, tasks), total=len(tasks))
    )
    pool.close()
    pool.join()
    return results


def dataset_split_continent(test_fold, val_fold):
    # Read your CSV file into a pandas DataFrame
    # Read your CSV file into a pandas DataFrame
    fold_path = "/Net/Groups/BGI/work_2/scratch/DeepExtremes/mc_earthnet_biome.csv"
    df = pd.read_csv(fold_path, delimiter=",")[
        ["path", "group", "check", "start_date", "lat", "lon", "biome"]
    ]

    # 3 training minicubes between 2017 - 2020 (randomly starting in the first year), followed by 3 test minicubes in 2021
    df["start_date"] = pd.to_datetime(df["start_date"], format="%Y-%m-%dT%H:%M:%S.%f")
    df["start_date2"] = df["start_date"] + datetime.timedelta(days=450)
    df["start_date3"] = df["start_date2"] + datetime.timedelta(days=450)
    df["start_test1"] = df["start_date3"] + datetime.timedelta(days=450)
    df["start_test2"] = df["start_test1"] + datetime.timedelta(days=90)
    df["start_test3"] = df["start_test2"] + datetime.timedelta(days=90)
    df = df.melt(
        ["path", "group", "check", "lat", "lon", "biome"], value_name="start_date"
    )
    df["end_date"] = df["start_date"] + datetime.timedelta(days=449)

    # Check if the point is in Africa
    def is_in_africa(row, africa):
        # Create a Shapely Point from latitude and longitude
        point = Point(row["lon"], row["lat"])

        # Check if the point is in Africa
        return africa.contains(point).any()

    # Load a world map dataset (included in geopandas)
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    africa = world[world["continent"] == "Africa"]
    # Apply the is_in_africa function to each row
    df["in_africa"] = df.apply(lambda row: is_in_africa(row, africa), axis=1)

    cols = ["path", "start_date", "end_date", "biome", "in_africa"]
    # temporal test set 2021
    temporal_test_subset = df.loc[
        (df["variable"].str.startswith("start_test")),
        cols,
    ]

    # folds 2017 - 2020
    df = df.loc[df["variable"].str.startswith("start_date")].drop("variable", 1)

    # training set
    train_subset = df.loc[
        (df["group"] != test_fold)
        & (df["group"] != val_fold)
        & (df["check"] == 0)
        & (df["in_africa"] == False),
        cols,
    ]

    # validation set
    val_subset = df.loc[
        (df["group"] == val_fold) & (df["check"] == 0) & (df["in_africa"] == False),
        cols,
    ]

    # iid test set
    spatial_test_subset = df.loc[
        (df["group"] == test_fold) & (df["check"] == 0) & (df["in_africa"] == False),
        cols,
    ]

    africa_test_subset = df.loc[
        (df["check"] == 0) & (df["in_africa"] == True),
        cols,
    ]

    return (
        train_subset,
        val_subset,
        spatial_test_subset,
        temporal_test_subset,
        africa_test_subset,
    )


if __name__ == "__main__":

    output_json = "/Net/Groups/BGI/scratch/crobin/PythonProjects/EarthNet/Data_analysis/ML4RS/results_trainset_ndvi.json"

    train_subset, _, _, _, _ = dataset_split_continent(10, 9)

    print("len of the dataset: ", train_subset.shape[0])

    # Call the function to process samples in parallel
    sample_statistics = process_samples_in_parallel(train_subset)
    # Transpose the results to get statistics for each variable
    # variable_statistics = list(map(list, zip(*sample_statistics)))
    # save the statistics for each variable
    data = []
    for sample in sample_statistics:
        (
            cubename,
            results,
        ) = sample
        sample_dict = {"path": cubename}
        sample_dict.update(results)
        data.append(sample_dict)

    with open(output_json, "w") as fp:
        json.dump(data, fp)

    # print(data)
    print("output written to: " + output_json)
