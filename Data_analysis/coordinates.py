from pathlib import Path
import datetime
import xarray as xr
import numpy as np
from tqdm import tqdm
from collections import Counter
import json

# Paths
basepath = Path("/scratch/crobin/earthnet2023_preprocessing/")
train_paths = list(basepath.glob("train/*/*.nc"))

print("len of the dataset: ", len(train_paths))

data = {}

for i, file in enumerate(tqdm(train_paths)):
    minicube = xr.open_dataset(file).load()

    longitude = np.mean(minicube.lon).item()
    latitude = np.mean(minicube.lat).item()
    time = minicube.time

    veg_type = Counter(minicube.esawc_lc.values.ravel()).most_common(1)[0][0]


    data[str(file)] = {"latitude": longitude, "longitude": latitude, "veg_type": veg_type, "time": time}

with open("Data_analysis/coordinates.json", "w") as fp:
    json.dump(data, fp)
