from pathlib import Path
import xarray as xr
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from scipy.interpolate import interp1d
import os


def interpolate_temporal(data):
    # Get the indices along the time_steps dimension
    time_indices = np.arange(len(data))

    # Create a mask for non-missing values
    mask = ~np.isnan(data)

    # If the first frame has missing values, fill them with the mean of the frame
    if mask.any():
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

def process_sample(file):
    minicube = xr.open_dataset(file, engine="netcdf4").load()
    step = [minicube.time.values[i] for i in range(4, 450, 5)]
    mask = minicube.s2_mask.sel(time=step).values < 1.0

    for variable in spatio_temporal_variables:
        # Get the variable data from the minicube
        var_data = minicube[variable].sel(time=step).values

        # Apply temporal interpolation to fill missing values
        var_data = np.where(var_data * mask == 0, np.nan, var_data)

        # Apply interpolation to all i, j positions simultaneously
        if ~mask.any():
            interpolated_results = np.apply_along_axis(
                interpolate_temporal, 0, var_data
            )

            # Combine the interpolated data back into the minicube
            minicube[variable].loc[dict(time=step)] = interpolated_results

    for variable in temporal_variables:
        var_data = minicube[variable].values
        interpolated_results = interpolate_temporal(var_data)
        minicube[variable].values = interpolated_results

    # Saving
    path = os.path.join(dst_path, str(file)[len(str(basepath)) + 1 : -13])
    if not os.path.exists(path):
        os.makedirs(path)
    name = os.path.join(path, str(file)[-12:])
    minicube.to_netcdf(name, engine="netcdf4")


if __name__ == "__main__":
    basepath = Path("/scratch/crobin/earthnet2023/train/")
    dst_path = "/scratch/crobin/earthnet2023_interpolated/train/"

    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    paths = list(basepath.glob("*/*.nc"))
    print("len of the dataset: ", len(paths))

    spatio_temporal_variables = [
        "s2_B02",
        "s2_B03",
        "s2_B04",
        "s2_B05",
        "s2_B06",
        "s2_B07",
        "s2_B8A",
        "s1_vv",
        "s1_vh",
    ]

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

    with mp.Pool(100) as pool:
        r = list(tqdm(pool.imap(process_sample, paths), total=len(paths)))
