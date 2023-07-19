from pathlib import Path
import datetime
import xarray as xr
import numpy as np
from tqdm import tqdm
import random
import os
import scipy as sy
import json



def interpolation(data, mask):
    def interp(x, y):
        xi = np.arange(data.shape[0])[mask[:, x, y]]
        yi =  data[:, x, y][mask[:, x, y]]
        interp = sy.interpolate.interp1d(xi, yi, kind = "linear")
        xj = np.arange(data.shape[0])[~mask[:, x, y]]
        data[:, x, y][~mask[:, x, y]] = interp(xj)
        return

    xx, yy = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[2]))

    data[0, :, :][~mask[0, :, :]] = np.mean(data[0, :, :][mask[0, :, :]])
    data[-1, :, :][~mask[-1, :, :]] = np.mean(data[-1, :, :][mask[-1, :, :]])
    mask[0,:,:] = mask[-1,:,:] = True
    vecinterp = np.vectorize(interp)
    vecinterp(xx, yy)
    return 

# Paths
basepath = Path("/scratch/crobin/earthnet2023/train/")
dst_path = "/scratch/crobin/earthnet2023_preprocessed/train/"
if not os.path.exists(dst_path):
    os.mkdirs(dst_path)

paths = list(basepath.glob("*/*.nc"))
print("len of the dataset: ", len(paths))

variables = ["s2_B02", "s2_B03", "s2_B04", "s2_B05", "s2_B06", "s2_B07", "s2_B8A","s2_avail", "s2_SCL","s2_mask", "s1_vv", "s1_vh", "s1_avail"]

for i, file in enumerate(tqdm(paths)):
    minicube = xr.open_dataset(file).load()

    step = [minicube.time.values[i] for i in range(4, 450, 5)]
    mask = (minicube.s2_mask.sel(time=step).values < 1.0)
    for variable in variables:
        data = minicube.variable.sel(time=step).values 
        interpolation(data, mask)
        minicube.variable.sel(time=step).values = data

    # Saving
    path = os.path.join(dst_path, str(file)[len(str(basepath))+1:-13])
    if not os.path.exists(path):
        os.mkdir(path)
    name = os.path.join(path, str(file)[-12:])
    minicube.to_netcdf(name)
    
# with open("Data_analysis/missing_value_test_from2016.json", "w") as fp:
 #   json.dump(data, fp)
