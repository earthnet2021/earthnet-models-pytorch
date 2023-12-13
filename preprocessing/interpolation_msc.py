from save_minicube import save_minicube_netcdf
import pandas as pd
from pathlib import Path
import datetime
import xarray as xr
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import os


def interpolate_msc(minicube):
    # Monthly mean seasonal cycle, time is the 15th of each month, of 1970 (fake years)
    msc_monthly = minicube.ndviclim_mean

    # Create a new DataArray 'msc' with extended time coordinates for interpolation
    time_start = minicube.time[0].values - np.timedelta64(20, "D")
    time_end = minicube.time[-1].values + np.timedelta64(20, "D")
    time = pd.date_range(start=time_start, end=time_end)
    msc = xr.DataArray(
        np.nan,
        coords={"time": time, "lat": minicube.lat, "lon": minicube.lon},
        dims=["time", "lat", "lon"],
    )

    # Attribute to msc the monthly values of msc_monthly)
    for month in range(1, 13):
        # Select the monthly values from 'msc_monthly'
        monthly_values = msc_monthly["time_clim.month" == month].values
        # Assign the monthly values from 'msc_monthly' to 'msc'
        msc[(msc["time.month"] == month) & (msc["time.day"] == 15)] = monthly_values

    # Interpolate the missing values of msc.
    msc_daily = msc.interpolate_na(dim="time")

    # Create a new DataArray 'msc' with the same coordinates as 'minicube' but filled with NaN values
    msc = xr.DataArray(
        np.nan,
        coords={"time": minicube.time, "lat": minicube.lat, "lon": minicube.lon},
        dims=["time", "lat", "lon"],
    )

    # Resample 'msc_daily' to match the specific 'time' subset
    time = [minicube.time.values[i] for i in range(4, 450, 5)]
    msc_daily_resampled = msc_daily.sel(time=time)

    # Assign the resampled values to 'msc' based on matching time values
    msc.loc[{"time": msc_daily_resampled["time"]}] = msc_daily_resampled.values
    return msc


def add_msc_daily_dataset(file):
    """Function to add a daily mean seasonal cycle dataset to an input file"""

    # Open the NetCDF dataset using the netCDF4 engine and load it into memory
    minicube = xr.open_dataset(file, engine="netcdf4").load()

    # Interpolate the mean seasonal cycle from monthly to daily
    msc = interpolate_msc(minicube)

    # Add the interpolated mean seasonal cycle as a variable to 'minicube'
    minicube["msc"] = msc

    # Add a description attribute to the 'msc' variable
    minicube["msc"].attrs[
        "description"
    ] = "Mean Seasonal Cycle interpolated daily from the monthly ndvi_mean variable."

    # Saving the modified dataset
    # Construct the path and filename for saving
    path = os.path.join(dst_path, str(file)[len(str(basepath)) + 1 : -13])
    name = os.path.join(path, str(file)[-12:])

    try:
        # Save the modified 'minicube' dataset as a NetCDF file
        save_minicube_netcdf(minicube, name)
    except:
        # Handle exceptions if an error occurs during saving
        print("An error occurred while saving the file: ", name)


if __name__ == "__main__":
    basepath = Path("/scratch/crobin/earthnet2023_interpolated_extremes/train/")
    dst_path = "/scratch/crobin/earthnet2023_preprocessing/train/"

    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    folders = list(basepath.glob("*"))
    for folder in folders:
        path = os.path.join(dst_path, str(folder)[len(str(basepath)) + 1 :])
        if not os.path.exists(path):
            os.makedirs(path)

    paths = list(basepath.glob("*/*.nc"))
    print("len of the dataset: ", len(paths))

    with mp.Pool(50) as pool:
        r = list(tqdm(pool.imap(add_msc_daily_dataset, paths), total=len(paths)))
