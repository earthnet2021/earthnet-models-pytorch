from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import multiprocessing as mp
import os


temporal_variables = [
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

def calculate_variable_statistics(data):
    """Function to calculate statistics for each variable in a sample"""

    max_value = np.nanmax(data)
    min_value = np.nanmin(data)
    sum_var = np.nansum(data)
    n_obs = np.count_nonzero(~np.isnan(data))
    return max_value, min_value, sum_var, n_obs

def calculate_nan(data):
    """Function to calculate statistics for each variable in a sample"""
    n_nan = np.count_nonzero(np.isnan(data))
    no_data = 1 if n_nan == data.shape[0] else 0
    return n_nan, no_data

def calculate_sample_statistics(file):
    minicube = xr.open_dataset(file, engine="zarr").load()
    return [calculate_variable_statistics(minicube[variable].values) for variable in temporal_variables]

def process_samples_in_parallel(paths):
    "Function to process samples using pdata = minicube[variable].valuesarallel processing"
    num_cores = mp.cpu_count()
    print("numbers of cores availables: ", num_cores)
    pool = mp.Pool(processes=100)
    results = list(tqdm(pool.imap(calculate_sample_statistics, paths), total=len(paths)))
    pool.close()
    pool.join()
    return results

def get_filepaths(metadata_file):
    # load csv
    df = pd.read_csv(metadata_file, delimiter=",")[["path", "check"]]
    return df.loc[ (df["check"] == 0) , ["path"]].values.tolist()


if __name__ == "__main__":
    basepath = "/Net/Groups/BGI/work_2/scratch/DeepExtremes/dx-minicubes"

    filepaths = get_filepaths("/Net/Groups/BGI/work_2/scratch/DeepExtremes/mc_earthnet.csv")

    paths = [Path(basepath + fp[0]) for fp in filepaths]

    print(paths[:10])
    print("len of the dataset: ", len(paths))

    # Call the function to process samples in parallel
    sample_statistics = process_samples_in_parallel(paths)

    # Transpose the results to get statistics for each variable
    variable_statistics = list(map(list, zip(*sample_statistics)))

    # save the statistics for each variable
    data = {}
    for var_idx, var_stats in enumerate(variable_statistics):
        var_name = (temporal_variables)[var_idx]
        max_vals, min_vals, sum_vals, n_obs_vals = zip(*var_stats)
        mean = np.sum(sum_vals) / np.sum(n_obs_vals)
        data[str(var_name)] = {"mean": np.float64(mean), "min": np.float64(np.nanmin(min_vals)), "max": np.float64(np.nanmax(max_vals))}
       
    with open("statistics_de23.json", "w") as fp:
        json.dump(data, fp)

