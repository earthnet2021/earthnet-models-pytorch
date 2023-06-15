from pathlib import Path
import datetime
import xarray as xr
import numpy as np
from tqdm import tqdm

# Paths
basepath = Path("/scratch/crobin/earthnet2023/")
train_paths = list(basepath.glob("train/*/*.nc"))

print("len of the dataset: ", len(train_paths))

missing_file = open("missing_values.txt", "w")
data = {}

for i, file in enumerate(tqdm(train_paths)):

