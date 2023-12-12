from pathlib import Path
import xarray as xr
import numpy as np
import json
from tqdm import tqdm
import multiprocessing as mp
import os


temporal_variables = [
    "era5land_t2m_mean",
    "era5land_pev_mean",
    "era5land_slhf_mean",
    "era5land_ssr_mean",
    "era5land_sp_mean",
    "era5land_sshf_mean",
    "era5land_e_mean",
    "era5land_tp_mean",
    "era5land_t2m_min",
    "era5land_pev_min",
    "era5land_slhf_min",
    "era5land_ssr_min",
    "era5land_sp_min",
    "era5land_sshf_min",
    "era5land_e_min",
    "era5land_tp_min",
    "era5land_t2m_max",
    "era5land_pev_max",
    "era5land_slhf_max",
    "era5land_ssr_max",
    "era5land_sp_max",
    "era5land_sshf_max",
    "era5land_e_max",
    "era5land_tp_max",
    "era5_e",
    "era5_pet",
    "era5_pev",
    "era5_ssrd",
    "era5_t2m",
    "era5_t2mmax",
    "era5_t2mmin",
    "era5_tp",
]

spatial_variables = [
    "sg_bdod_top_mean",
    "sg_bdod_sub_mean",
    "sg_cec_top_mean",
    "sg_cec_sub_mean",
    "sg_cfvo_top_mean",
    "sg_cfvo_sub_mean",
    "sg_clay_top_mean",
    "sg_clay_sub_mean",
    "sg_nitrogen_top_mean",
    "sg_nitrogen_sub_mean",
    "sg_phh2o_top_mean",
    "sg_phh2o_sub_mean",
    "sg_ocd_top_mean",
    "sg_ocd_sub_mean",
    "sg_sand_top_mean",
    "sg_sand_sub_mean",
    "sg_silt_top_mean",
    "sg_silt_sub_mean",
    "sg_soc_top_mean",
    "sg_soc_sub_mean",
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
    minicube = xr.open_dataset(file, engine="netcdf4").load()
    return [calculate_variable_statistics(minicube[variable].values) for variable in temporal_variables + spatial_variables]


def process_samples_in_parallel(paths):
    "Function to process samples using pdata = minicube[variable].valuesarallel processing"
    num_cores = mp.cpu_count()
    print("numbers of cores availables: ", num_cores)
    pool = mp.Pool(processes=100)
    results = list(tqdm(pool.imap(calculate_sample_statistics, paths), total=len(paths)))
    pool.close()
    pool.join()
    return results



if __name__ == "__main__":
    basepath = Path("/scratch/crobin/earthnet2023/train/")

    paths = list(basepath.glob("*/*.nc"))
    print("len of the dataset: ", len(paths))

    # Call the function to process samples in parallel
    sample_statistics = process_samples_in_parallel(paths)

    # Transpose the results to get statistics for each variable
    variable_statistics = list(map(list, zip(*sample_statistics)))

    # save the statistics for each variable
    data = {}
    for var_idx, var_stats in enumerate(variable_statistics):
        var_name = (temporal_variables + spatial_variables)[var_idx]
        max_vals, min_vals, sum_vals, n_obs_vals = zip(*var_stats)
        mean = np.sum(sum_vals) / np.sum(n_obs_vals)
        data[str(var_name)] = {"mean": np.float64(mean), "min": np.float64(np.nanmin(min_vals)), "max": np.float64(np.nanmax(max_vals))}
       
    #for var_idx, var_stats in enumerate(variable_statistics):
    #    var_name = (temporal_variables + spatial_variables)[var_idx]
    #    n_nan_vals, no_data_vals = zip(*var_stats)
    #    mean = np.sum(n_nan_vals) / (450 * len(paths))
    #    no_data = np.sum(no_data_vals) / len(paths)
    #
    #    data[str(var_name)] = {"mean": np.float64(mean), "no_data": no_data}
        
    with open("statistics_en23.json", "w") as fp:
        json.dump(data, fp)