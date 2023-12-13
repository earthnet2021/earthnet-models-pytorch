from save_minicube import save_minicube_netcdf
from pathlib import Path
import xarray as xr
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import os


def add_extremes_events_dataset(file):
    """function to add extremes events dataset to input files"""

    # Define a helper function to convert longitude values to coordinates
    def longitudetocoords(x):
        return (((x - 180) % 360) + 180) % 360

    # Define the path to the extremes events data in Zarr format
    path2cube = "/Net/Groups/BGI/work_1/scratch/s3/deepextremes/v2/EventCube_ranked_pot0.01_ne0.1.zarr"
    
    # Open the Zarr dataset
    data = xr.open_zarr(path2cube)

    # Open the input dataset in NetCDF4 format and load it into memory
    minicube = xr.open_dataset(file, engine="netcdf4").load()

    # Calculate the mean longitude, latitude, and time from the input dataset
    longitude = np.mean(minicube.lon).item()
    latitude = np.mean(minicube.lat).item()
    time = minicube.time

    # Select extremes events data based on the nearest longitude, latitude, and time
    extremes = data.sel(
        longitude=longitudetocoords(longitude), latitude=latitude, method="nearest"
    ).sel(time=time)
    extremes.drop_vars('longitude')
    extremes.drop_vars('latitude')

    # Add the 'extremes' as a variable to 'minicube'
    extremes = xr.DataArray(extremes.layer.values, coords={'time': minicube.time}, dims=['time'])
    minicube["extremes"] = extremes

    # Add a description attribute to the 'extremes' variable
    minicube["extremes"].attrs[
        "description"
    ] = "extremes is Int8 values coding for types of Discrete Extreme Occurrences. The first (least significant) bit is set to 1 where P(X~Tmax) > 0.99. The second to fourth bits are set to 1 where P(X~PEI) < 0.01, for PEI~30, PEI~90 and PEI~180 respectively. The fifth bit is for non extremes, i.e. where all four indicators lie in the middle of their distribution (P(X) > 0.1 AND P(X) < 0.9). More information on https://github.com/DeepExtremes/large-cube-access/blob/main/data_access_python.ipynb"

    # Save the modified dataset as a NetCDF file
    path = os.path.join(dst_path, str(file)[len(str(basepath)) + 1 : -13])
    name = os.path.join(path, str(file)[-12:])
    
    try:
        save_minicube_netcdf(minicube, name)
    except:
        print("An error occurred while saving the file: ", name)

if __name__ == "__main__":
    # Define the base path for input files and the destination path for output files
    basepath = Path("/Net/Groups/BGI/work_1/scratch/DeepCube/earthnet2023_interpolated/train")
    dst_path = "/scratch/crobin/earthnet2023_interpolated_extremes/train/"

    # Create the destination directory if it doesn't exist
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    # Create directories for subfolders within the destination path
    folders = list(basepath.glob("*"))
    for folder in folders:
        path = os.path.join(dst_path, str(folder)[len(str(basepath)) + 1 :])
        if not os.path.exists(path):
            os.makedirs(path)

    # Get a list of input file paths
    paths = list(basepath.glob("*/*.nc"))
    print("len of the dataset: ", len(paths))

    # Use multiprocessing to run the 'add_extremes_events_dataset' function on input files
    with mp.Pool(20) as pool:
        r = list(tqdm(pool.imap(add_extremes_events_dataset, paths), total=len(paths)))