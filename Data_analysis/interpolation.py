from pathlib import Path
import datetime
import xarray as xr
import numpy as np
from tqdm import tqdm
import random
import multiprocessing as mp
from scipy.interpolate import interp1d
import os


def compute_scale_and_offset(savepath, v, da, n=16):
    """From earthnet-minicuber. Calculate offset and scale factor for int conversion.

    Based on Krios101's code above.
    """
    if ~np.isnan(da).all():
        vmin = np.nanmin(da).item()
        vmax = np.nanmax(da).item()
    else:
        vmin = np.nan
        vmax = np.nan

    # stretch/compress data to the available packed range
    scale_factor = (vmax - vmin) / (2**n - 1)

    # translate the range to be symmetric about zero
    add_offset = vmin + 2 ** (n - 1) * scale_factor

    return scale_factor, add_offset


@staticmethod
def save_minicube_netcdf(minicube, savepath):
    """From earthnet-minicuber."""
    savepath = Path(savepath)
    encoding = {}
    for v in list(minicube.variables):
        if v in ["time", "time_clim", "lat", "lon"]:
            continue

        elif ("interpolation_type" in minicube[v].attrs) and (
            minicube[v].attrs["interpolation_type"] == "linear"
        ):
            scale_factor, add_offset = compute_scale_and_offset(
                savepath, v, minicube[v].values
            )
        else:
            scale_factor, add_offset = 1.0, 0.0

        if (
            abs(scale_factor) < 1e-8
            or np.isnan(scale_factor)
            or (scale_factor == 1.0 and minicube[v].max() > 32766)
        ):
            encoding[v] = {"zlib": True, "complevel": 9}

        else:
            encoding[v] = {
                "dtype": "int16",
                "scale_factor": scale_factor,
                "add_offset": add_offset,
                "_FillValue": -32767,
                "zlib": True,
                "complevel": 9,
            }

    if savepath.is_file():
        savepath.unlink()
    else:
        savepath.parents[0].mkdir(exist_ok=True, parents=True)

    minicube.to_netcdf(savepath, encoding=encoding, compute=True)


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

    # reshape test set for 450 days.
    if len(minicube.time.dt.date.values) > 450:
        minicube[spatio_temporal_variables] = minicube[spatio_temporal_variables].where(
            minicube.time.dt.date > datetime.date(2017, 6, 30), drop=True
        )

        if len(minicube.time.dt.date.values) > 0:
            # Subset of every 5 days
            indexes_avail = np.where(minicube.s2_avail.values == 1)[0]
            if len(indexes_avail) > 0:
                # Random subset of size 450 days.
                beg = random.choice(indexes_avail[1:-90])
                minicube = minicube.isel(time=slice(beg - 4, beg - 4 + 450))

                indexes_avail = np.where(minicube.s2_avail.values == 1)[0]
                step = [minicube.time.values[i] for i in range(4, 450, 5)]
                dates = [minicube.time.values[i] for i in (indexes_avail)]

                # Condition to check that the every 5 days inclus all the dates available (+ missing day)
                if not set(dates) <= set(step):  # dates is a subset of time.
                    print("ERROR: ", file)

    step = [minicube.time.values[i] for i in range(4, 450, 5)]
    mask = minicube.s2_mask.sel(time=step).values < 1.0

    for variable in spatio_temporal_variables:
        # Get the variable data from the minicube
        var_data = minicube[variable].sel(time=step).values

        # Apply temporal interpolation to fill missing values
        var_data = np.where(var_data * mask == 0, np.nan, var_data)
        # Apply interpolation to all i, j positions simultaneously
        if mask.any():
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
    name = os.path.join(path, str(file)[-12:])

    try:
        save_minicube_netcdf(minicube, name)
    except:
        print("An error occured to save the file: ", name)


#
if __name__ == "__main__":
    basepath = Path("/scratch/crobin/earthnet2023/train/")
    dst_path = "/scratch/crobin/earthnet2023_interpolated/train/"

    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    folders = list(basepath.glob("*"))
    for folder in folders:
        path = os.path.join(dst_path, str(folder)[len(str(basepath)) + 1 :])
        if not os.path.exists(path):
            os.makedirs(path)

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

    with mp.Pool(50) as pool:
        r = list(tqdm(pool.imap(process_sample, paths), total=len(paths)))
