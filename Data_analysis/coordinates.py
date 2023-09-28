from pathlib import Path
import xarray as xr
import numpy as np
from tqdm import tqdm
from collections import Counter
import pandas as pd
import sys

# Paths
basepath = Path("/scratch/crobin/earthnet2023_preprocessing")  # Update with your actual base path
train_paths = list(basepath.glob("train/*/*.nc"))[:5]

print("len of the dataset: ", len(train_paths))

# Create an empty dataset with the desired coordinates
lon = np.arange(-20.00, 56.00, 0.01)
lat = np.arange(38.00, -36.00, -0.01)
time = pd.date_range(start="2015-01-01", end="2022-12-31", freq="D")
data = xr.Dataset(
    {
        "veg_type": xr.DataArray(
            data=np.empty((len(lon), len(lat)), dtype=bytes),
            coords={"lon": lon, "lat": lat},
            dims=["lon", "lat"],
            attrs={"description": "vegetation type of the minicube"},
        ), #.chunk({'lat': 25, 'lon': 25})
        "variable": xr.DataArray(
            data=np.empty((len(lon), len(lat), len(time)), dtype=bytes),
            coords={"lon": lon, "lat": lat, "time": time},
            dims=["lon", "lat", "time"],
            attrs={"description": "minicube distribution"},
        )
    },
    attrs={"description": "earthnet2023 dataset"},
)

for i, file in enumerate(tqdm(train_paths)):
    minicube = xr.open_dataset(file).load()

    longitude = np.mean(minicube.lon).item()
    latitude = np.mean(minicube.lat).item()
    time = minicube.time.values  # Extract time values as a numpy array
    veg_type = Counter(minicube.esawc_lc.values.ravel()).most_common(1)[0][0]
    
    # Assign veg_type to the corresponding position in data
    data["veg_type"].sel(lon =longitude, lat=latitude, method = 'nearest').values = veg_type
    data["variable"].sel(lon =longitude, lat=latitude, time=time, method = 'nearest').values = np.ones(time.shape)

# Save the dataset to Zarr
data.to_zarr("earthnet2023_veg_type.zarr")

