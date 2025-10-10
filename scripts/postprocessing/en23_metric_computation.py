from pathlib import Path
import xarray as xr
import numpy as np
import json
from tqdm import tqdm
import multiprocessing as mp
import os
from scipy.stats import pearsonr


def calculate_variable_statistics(minicube, pred, anomalie=False, subset="all"):
    """Function to calculate statistics for each variable in a sample"""
    time = [minicube.time.values[i] for i in range(4, 450, 5)]
    context_length = 60
    target_length = 10
    target = (minicube.s2_B8A - minicube.s2_B04) / (
        minicube.s2_B8A + minicube.s2_B04 + 1e-8
    )
    if anomalie:
        target = target - minicube.msc
    targ = target.sel(time=time)[context_length : context_length + target_length, ...]
    pred = pred.ndvi_pred

    extremes = minicube.extremes[
        context_length * 5 : (context_length + target_length) * 5, ...
    ]
    if np.logical_and(1 <= extremes, extremes <= 15).any():
        extreme = True
    else:
        extreme = False

    cloud_mask = (
        minicube.s2_mask.sel(time=time)[
            context_length : context_length + target_length, ...
        ]
        < 1.0
    )
    lc = minicube.esawc_lc

    lc_mask = (lc <= 40) | (lc >= 90)
    lc_mask = np.expand_dims(lc_mask, axis=0)
    lc_mask = np.repeat(lc_mask, 10, axis=0)

    mask = cloud_mask * lc_mask

    targ = np.where(targ * mask == 0, np.nan, targ)
    pred = np.where(pred * mask == 0, np.nan, pred)
    n_obs = np.count_nonzero(~np.isnan(targ), axis=0)  # per pixel in space
    if (
        (np.sum(n_obs) / (pred.shape[0] * pred.shape[1] * pred.shape[2]) < 0.1)
        or (subset == "extreme" and not extreme)
        or (subset == "non_extreme" and extreme)
    ):
        return None, None, None, None

    else:
        lc_mask = (lc <= 40) | (lc >= 90)
        landcover = np.where(lc * lc_mask == 0, np.nan, lc)
        # Use numpy's unique function to get unique elements and their counts
        unique_elements, counts = np.unique(
            landcover[~np.isnan(landcover)].flatten(), return_counts=True
        )

        # Find the index of the maximum count

        max_count_index = np.argmax(counts)

        # The most common number
        lc_maj = unique_elements[max_count_index]

        sum_squared_error = np.nansum((targ - pred) ** 2, axis=0)

        mse = sum_squared_error / (n_obs + 1e-8)

        r2 = np.zeros((128, 128))
        sum_squared_dev = np.zeros((128, 128))
        for i in range(128):
            for j in range(128):
                x = targ[:, i, j]
                y = pred[:, i, j]

                nas = np.logical_or(np.isnan(x), np.isnan(y))
                if (
                    len(x[~nas]) > 1
                    and ~np.all(x[~nas] == x[~nas][0])
                    and ~np.all(y[~nas] == y[~nas][0])
                ):
                    x = x[~nas]
                    y = y[~nas]

                    mean_targ = np.mean(x)
                    mean_pred = np.mean(y)

                    dev_targ = x - mean_targ
                    dev_pred = y - mean_pred

                    dev_targ_squared = dev_targ**2
                    dev_pred_squared = dev_pred**2

                    r = np.sum(dev_targ * dev_pred) / (
                        np.sqrt(np.sum(dev_pred_squared))
                        * np.sqrt(np.sum(dev_targ_squared))
                        + 1e-8
                    )
                    r2[i, j] = r**2
                    sum_squared_dev[i, j] = np.sum(dev_targ_squared)

                else:
                    r2[i, j] = np.nan
                    sum_squared_dev[i, j] = np.nan

        nse = 1 - (sum_squared_error / (sum_squared_dev + 1e-8))
        mse = np.nanmean(mse) if np.count_nonzero(~np.isnan(mse)) != 0 else None
        nse = np.nanmedian(nse) if np.count_nonzero(~np.isnan(nse)) != 0 else None
        r2 = np.nanmean(r2) if np.count_nonzero(~np.isnan(r2)) != 0 else None
        return mse, nse, r2, lc_maj


def calculate_sample_statistics(pred_path):
    name = str(pred_path)[-12:]
    test_path = list(
        Path("/scratch/crobin/earthnet2023_preprocessing/test/").glob("*/" + name)
    )[0]
    pred_anomalie_path = list(
        Path(
            "/Net/Groups/BGI/scratch/crobin/PythonProjects/EarthNet/earthnet-models-pytorch/experiments/en23/convlstm_ae/convlstm_ae/config_16.10.2023_anomalie_NDVI/preds/iid"
        ).glob("*/" + name)
    )[0]
    target = xr.open_dataset(test_path, engine="netcdf4").load()
    pred = xr.open_dataset(pred_path, engine="netcdf4").load()
    pred_anomalie = xr.open_dataset(pred_anomalie_path, engine="netcdf4").load()

    return [
        calculate_variable_statistics(target, prediction, anomalie, "non_extreme")
        for prediction, anomalie in [(pred, False), (pred_anomalie, True)]
    ]


def process_samples_in_parallel(paths):
    "Function to process samples using parallel processing"
    num_cores = mp.cpu_count()
    print("numbers of cores availables: ", num_cores)
    pool = mp.Pool(processes=100)
    results = list(
        tqdm(pool.imap(calculate_sample_statistics, paths), total=len(paths))
    )
    pool.close()
    pool.join()
    return results


if __name__ == "__main__":
    basepath = Path(
        "/Net/Groups/BGI/scratch/crobin/PythonProjects/EarthNet/earthnet-models-pytorch/experiments/en23/convlstm_ae/convlstm_ae/config_16.10.2023_absolute_NDVI/preds/iid/"
    )

    paths = list(basepath.glob("*/*.nc"))
    print("len of the dataset: ", len(paths))

    # Call the function to process samples in parallel
    sample_statistics = process_samples_in_parallel(paths)

    # Transpose the results to get statistics for each variable
    variable_statistics = list(map(list, zip(*sample_statistics)))

    # save the statistics for each variable
    data, total = {}, {}

    for var_idx, var_stats in enumerate(variable_statistics):
        var_name = ["absolute_ndvi", "anomalie_ndvi"][var_idx]

        mse, nse, r2, lc_maj = zip(*var_stats)
        mse = np.float64(mse).tolist()
        nse = np.float64(nse).tolist()
        r2 = np.float64(r2).tolist()
        lc_maj = np.float64(lc_maj)

        data[str(var_name)] = {
            "rmse": [np.sqrt(i) for i in mse if i is not (None or np.nan)],
            "nse": [i for i in nse if i is not (None or np.nan)],
            "r2": [i for i in r2 if i is not (None or np.nan)],
            "lc_maj": [i for i in lc_maj if i is not (None or np.nan)],
        }
        total[str(var_name)] = {
            "rmse": np.sqrt(np.nanmean(mse)),
            "nse": np.nanmedian(nse),
            "r2": np.nanmean(r2),
        }
    name_file = "results_test.json"
    with open("results_test_non_extreme_final.json", "w") as fp:
        json.dump(data, fp)

    with open("total_extreme_" + name_file, "w") as fp:
        json.dump(total, fp)
