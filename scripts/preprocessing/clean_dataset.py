from pathlib import Path
import xarray as xr
import numpy as np
import json
import datetime
from tqdm import tqdm
import multiprocessing as mp
import os
from scipy.stats import pearsonr
from argparse import ArgumentParser
import sys
import pandas as pd



def is_valide(
    raw
):
    """Function to calculate statistics for each variable in a sample"""
    filepath = raw['filepath']
    t0 = pred.time.values[0]
    target_len = 13
    # time start index
    it = np.where(target.time == t0)[0][0]
    # first target_len steps, every 5 days
    tind = range(it, it + target_len, 1)

    targ = (target.B8A - target.B04) / (target.B8A + target.B04 + 1e-8)[tind, ...]

    # set mask = 1 where data are OK
    cloud_mask = target.cloudmask_en[tind, ...] == 0
    lc = target.SCL[tind, ...]
    lc_mask = (lc == 4) | (lc == 4)
    mask = cloud_mask * lc_mask

    targ = np.where(targ * mask == 0, np.nan, targ)
    n_obs = np.count_nonzero(~np.isnan(targ), axis=0)
    if np.sum(n_obs) / (128 * 128 * target_len) > 0.05:
        return True
    else:
        return False


if __name__ == "__main__":
    # Read your CSV file
    df = pd.read_csv(
        "/Net/Groups/BGI/work_2/scratch/DeepExtremes/mc_earthnet_biome.csv",
        delimiter=",",
    )
    df["start_date"] = pd.to_datetime(
            df["start_date"], format="%Y-%m-%dT%H:%M:%S.%f"
        )
    df["start_date2"] = df["start_date"] + datetime.timedelta(days=450)
    df["start_date3"] = df["start_date2"] + datetime.timedelta(days=450)
    df["start_test1"] = df["start_date3"] + datetime.timedelta(days=450)
    df["start_test2"] = df["start_test1"] + datetime.timedelta(days=90)
    df["start_test3"] = df["start_test2"] + datetime.timedelta(days=90)
    df = df.melt(["path", "group", "check", "lat", "lon"], value_name="start_date")
    df["end_date"] = df["start_date"] + datetime.timedelta(days=449)

    paths = list(
        Path("/Net/Groups/BGI/tscratch/mweynants/dx-minicubes").glob("full/*/*.zarr")
    )[0]
    print("len paths: ", len(paths))
    output = "/Net/Groups/BGI/work_2/scratch/DeepExtremes/mc_earthnet_biome.csv")

    df["is_valid"] = df.apply(lambda row: is_valid(row), axis=1)
    df.to_csv(output)
