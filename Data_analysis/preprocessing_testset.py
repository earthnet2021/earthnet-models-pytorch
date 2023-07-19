from pathlib import Path
import datetime
import xarray as xr
import numpy as np
from tqdm import tqdm
import json
import random
import os
import json

# Paths
basepath = Path("/scratch/crobin/earthnet2023/test_original/")
dst_path = "/scratch/crobin/earthnet2023/test/"
if not os.path.exists(dst_path):
                os.mkdir(dst_path)

paths = list(basepath.glob("*/*.nc"))
print("len of the dataset: ", len(paths))

variables = ["s2_B02", "s2_B03", "s2_B04", "s2_B05", "s2_B06", "s2_B07", "s2_B8A","s2_avail", "s2_SCL","s2_mask", "s1_vv", "s1_vh", "s1_avail"]

for i, file in enumerate(tqdm(paths)):
    minicube = xr.open_dataset(file).load()

    # Subset of period with the 2 satellite and only 5 days gap
    minicube[variables] = minicube[variables].where(
        minicube.time.dt.date > datetime.date(2017, 6, 30), drop=True
    )
    if len(minicube.time.dt.date.values) > 0:
        # Subset of every 5 days
        indexes_avail = np.where(minicube.s2_avail.values == 1)[0]
        if len(indexes_avail) > 0:
        # Random subset of size 450 days.
            beg = random.choice(indexes_avail[1:-90])
            minicube  = minicube.isel(time=slice(beg - 4, beg - 4 + 450))

            indexes_avail = np.where(minicube.s2_avail.values == 1)[0]
            time = [
                minicube.time.values[i]
                for i in range(4, 450, 5)
            ]
            dates = [minicube.time.values[i] for i in (indexes_avail)]

            # Condition to check that the every 5 days inclus all the dates available (+ missing day)
            if set(dates) <= set(time): # dates is a subset of time.
                mask = minicube.s2_mask.sel(time=time)
            else:
                print("ERROR: ", file)

            # # Transform
            # mask.values[mask.values > 0] = np.nan
    # 
            # # Computation of the NaN values
            # total_missing = np.sum(np.isnan(mask), axis=(0, 1, 2)).values.tolist()
            # serie_missing = np.sum(np.isnan(mask), axis=(1, 2)).values.tolist()
            # data[str(file)] = {"total_missing": total_missing, "serie_missing": serie_missing}
    # 
            # if total_missing / (len(serie_missing) * 128 * 128) < 0.25:
            path = os.path.join(dst_path, str(file)[len(str(basepath))+1:-13])
            if not os.path.exists(path):
                os.mkdir(path)
            name = os.path.join(path, str(file)[-12:])
            minicube.to_netcdf(name)
    
# with open("Data_analysis/missing_value_test_from2016.json", "w") as fp:
 #   json.dump(data, fp)
