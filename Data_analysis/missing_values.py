from pathlib import Path
import datetime
import xarray as xr
import numpy as np
from tqdm import tqdm
import json

# Paths
basepath = Path("/Net/Groups/BGI/work_1/scratch/s3/earthnet/earthnet2023/")
train_paths = list(basepath.glob("test/*/*.nc"))

print("len of the dataset: ", len(train_paths))

data = {}

for i, file in enumerate(tqdm(train_paths)):
    minicube = xr.open_dataset(file).load()
    # Subset of period with the 2 satellite and only 5 days gap
    minicube = minicube.where(
        minicube.time.dt.date > datetime.date(2017, 6, 30), drop=True
    )
    if len(minicube.time.dt.date.values) > 0:
        # Subset of every 5 days
        indexes_avail = np.where(minicube.s2_avail.values == 1)[0]
        if len(indexes_avail) > 0:
            time = [
                minicube.time.values[i]
                for i in range(indexes_avail[0], indexes_avail[-1] + 1, 5)
            ]
            dates = [minicube.time.values[i] for i in (indexes_avail)]
            # Condition to check that the every 5 days inclus all the dates available (+ missing day)
            if set(dates) <= set(time):
                mask = minicube.s2_mask.sel(time=time)
            else:
                print("ERROR: ", file)

            # Transform
            mask.values[mask.values > 0] = np.nan

            # Computation of the NaN values
            s = np.sum(np.isnan(mask), axis=(0, 1, 2)).values.tolist()
            serie_nan = np.sum(np.isnan(mask), axis=(1, 2)).values.tolist()
            data[str(file)] = {"total_missing": s, "serie_missing": serie_nan}


with open("Data_analysis/missing_value_test_from2017.json", "w") as fp:
    json.dump(data, fp)

