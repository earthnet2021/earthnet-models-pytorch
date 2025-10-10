from pathlib import Path
import xarray as xr
import numpy as np
import json
from tqdm import tqdm
import multiprocessing as mp
import os
from scipy.stats import pearsonr
from argparse import ArgumentParser
import sys

era5 = [
    "t2m_mean",
    "pev_mean",
    "slhf_mean",
    "ssr_mean",
    "sp_mean",
    "sshf_mean",
    "e_mean",
    "tp_mean",
    "t2m_min",
    "pev_min",
    "slhf_min",
    "ssr_min",
    "sp_min",
    "sshf_min",
    "e_min",
    "tp_min",
    "t2m_max",
    "pev_max",
    "slhf_max",
    "ssr_max",
    "sp_max",
    "sshf_max",
    "e_max",
    "tp_max",
]


def calculate_variable_statistics(
    target,
    pred,
    cubename="",
    method="by_frame",
    season="DJF",
):
    """Function to calculate statistics for each variable in a sample"""
    t0 = pred.time.values[0]
    target_len = len(pred.time)  # - 1 fixing mistake len prediction iid test set...
    # time start index
    it = np.where(target.time == t0)[0][0]
    # first target_len steps, every 5 days
    tind = range(it, it + target_len, 1)

    targ = (target.B8A - target.B04) / (target.B8A + target.B04 + 1e-8)[tind, ...]
    pred = pred.ndvi_pred  # [:-1, ...]

    # set mask = 1 where data are OK
    cloud_mask = target.cloudmask_en[tind, ...] == 0
    lc = target.SCL[tind, ...]
    lc_mask = lc == 4
    mask = cloud_mask * lc_mask

    targ = np.where(targ * mask == 0, np.nan, targ)
    pred = np.where(pred * mask == 0, np.nan, pred)
    n_obs = np.count_nonzero(~np.isnan(targ), axis=0)
    if np.sum(n_obs) / (128 * 128 * target_len) < 0.05:
        # not enough data
        return (
            cubename,
            season,
            {},
            round(np.sum(n_obs) / (128 * 128 * target_len) * 100, ndigits=3),
        )
    # Meteorological variables
    era5_stats = {}
    for variable in era5:
        if variable[-3:] == "min":
            era5_stats[variable] = np.float64(np.nanmin(target[variable][tind]))
        elif variable[-3:] == "max":
            era5_stats[variable] = np.float64(np.nanmax(target[variable][tind]))
        else:
            era5_stats[variable] = np.float64(np.nanmean(target[variable][tind]))

    # NDVI
    # mean_ndvi = np.nansum(targ) / n_obs
    # std_ndvi = np.sqrt(np.nansum(targ - mean_ndvi))
    if method == "by_frame":
        # stats are first computed over time and then averaged over space by minicube
        sum_squared_error = np.nansum((targ - pred) ** 2, axis=0)
        rmse = np.sqrt(sum_squared_error / (n_obs + 1e-8))

        r2 = np.zeros((128, 128))
        mean_targ = np.zeros((128, 128))
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

                    mean = np.mean(x)
                    dev_targ = x - mean

                    dev_targ_squared = dev_targ**2

                    mean_targ[i, j] = mean
                    r2[i, j] = (pearsonr(x, y).statistic) ** 2
                    sum_squared_dev[i, j] = np.sum(dev_targ_squared)
                else:
                    sum_squared_dev[i, j] = np.nan
                    r2[i, j] = np.nan

        nse = 1 - (sum_squared_error / (sum_squared_dev + 1e-8))
        nnse = 1 / (2 - nse)
        rmse = np.nanmean(rmse) if np.count_nonzero(~np.isnan(rmse)) != 0 else None
        nnse = np.nanmean(nnse) if np.count_nonzero(~np.isnan(nnse)) != 0 else None
        nse = np.nanmedian(nse) if np.count_nonzero(~np.isnan(nse)) != 0 else None
        r2 = np.nanmean(r2) if np.count_nonzero(~np.isnan(r2)) != 0 else None
        mean_targ = (
            np.nanmean(mean_targ)
            if np.count_nonzero(~np.isnan(mean_targ)) != 0
            else None
        )
        sum_squared_dev = (
            np.nanmean(sum_squared_dev)
            if np.count_nonzero(~np.isnan(sum_squared_dev)) != 0
            else None
        )

        result = {
            "rmse": np.float64(rmse),
            "nse": np.float64(nse),
            "nnse": np.float64(nnse),
            "r2": np.float64(r2),
            "mean_targ": np.float64(mean_targ),
            "sigma_squared_targ": np.float64(sum_squared_dev),
        }
        result.update(era5_stats)

        return (
            cubename,
            season,
            result,
            round(np.sum(n_obs) / (128 * 128 * target_len) * 100, ndigits=3),
        )
    elif method == "overall":
        sum_squared_error = np.nansum((targ - pred) ** 2)
        rmse = np.sqrt(sum_squared_error / (np.sum(n_obs) + 1e-8))
        nas = np.logical_or(np.isnan(targ), np.isnan(pred))

        r2 = (pearsonr(targ[~nas], pred[~nas]).statistic) ** 2
        sum_squared_dev = np.nansum((targ - np.nanmean(targ)) ** 2)
        nse = 1 - (sum_squared_error / (sum_squared_dev + 1e-8))

        result = {
            "rmse": np.float64(rmse),
            "nse": np.float64(nse),
            "r2": np.float64(r2),
        }

        return (
            cubename,
            season,
            result,
            round(np.sum(n_obs) / (128 * 128 * target_len) * 100, ndigits=3),
        )
    else:
        raise NameError("Method " + method + " is not defined.")


def get_season(date: np.datetime64) -> str:
    # 1: MAM, 2: JJA, 3: SON, 4: DJF
    seasons = [
        "MAM",
        "JJA",
        "SON",
        "DJF",
    ]
    season = int(str(np.datetime64(date, "M"))[5:]) // 3 - 1
    return seasons[3 if season < 0 else season]


def get_name(path: Path) -> str:
    """Helper function gets Cubename from a Path

    Args:
        path (Path): One of Path/to/cubename.npz and Path/to/experiment_cubename.npz

    Returns:
        [str]: cubename (has format mc_lon_lat)
    """
    components = path.name.split("/")
    cubename_target = components[-1][0:-16]
    cubename_pred = components[-1]
    season = get_season(np.datetime64(components[-1][-15:-5]))
    return cubename_target, cubename_pred, season


def calculate_sample_statistics(pred_path, method="by_frame"):
    cubename_target, cubename_pred, season = get_name(pred_path)
    # try:
    pred = xr.open_dataset(pred_path, engine="zarr").load()
    # get test paths
    test_path = list(
        Path("/Net/Groups/BGI/tscratch/mweynants/dx-minicubes").glob(
            "full/*/" + cubename_target + "*.zarr"
        )
    )[0]
    target = xr.open_dataset(test_path, engine="zarr").load()
    return calculate_variable_statistics(target, pred, cubename_pred, method, season)


def calculate(args):
    result = calculate_sample_statistics(*args)
    return result


def process_samples_in_parallel(paths, method="by_frame"):
    "Function to process samples using parallel processing"
    num_cores = mp.cpu_count()
    print("numbers of cores availables: ", num_cores)
    pool = mp.Pool(processes=100)
    tasks = [(path, method) for path in paths]
    results = list(
        # tqdm(pool.imap(calculate_sample_statistics, paths,), total=len(paths))
        tqdm(pool.imap(calculate, tasks), total=len(tasks))
    )
    pool.close()
    pool.join()
    return results


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "bpath",
        type=str,
        metavar="path/to/experiment",
        help="experiments results",
    )
    # bpath = Path(
    #     "/Net/Groups/BGI/scratch/mweynants/DeepExtremes/earthnet-models-pytorch/experiments/de23/convlstm_ae/convlstm_ae/config_lr_1e-4_mlst50"
    # )
    parser.add_argument(
        "output",
        type=str,
        metavar="path/to/output",
        help="path to save output json file",
    )

    parser.add_argument(
        "method",
        type=str,
        metavar="by_frame",
        help="metrics computation method. 'by_frame' or 'overall'",
    )

    parser.add_argument(
        "test_set",
        type=str,
        metavar="iid",
    )

    args = parser.parse_args()

    path_list = args.bpath.split("/")[:-2]
    bpath = "/".join(path_list)

    components = bpath.split("/")
    output_json = (
        "results_"
        + components[-4]
        + "_"
        + components[-2]
        + "_"
        + components[-1]
        + "_"
        + args.test_set
        + "_"
        + args.method
        + ".json"
    )
    print(Path(args.output, output_json))

    basepath = Path(bpath, "preds/" + args.test_set + "/")

    # paths to test predictions
    paths = list(basepath.glob("*.zarr"))

    print("len of the dataset: ", len(paths))

    # Call the function to process samples in parallel
    sample_statistics = process_samples_in_parallel(paths, method=args.method)
    # Transpose the results to get statistics for each variable
    # variable_statistics = list(map(list, zip(*sample_statistics)))
    # save the statistics for each variable
    data = []
    for sample in sample_statistics:
        (
            cubename,
            season,
            results,
            valid_pixels,
        ) = sample
        sample_dict = {"name": cubename, "season": season, "valid_pixels": valid_pixels}
        sample_dict.update(results)
        data.append(sample_dict)

    with open(Path(args.output, output_json), "w") as fp:
        json.dump(data, fp)

    # print(data)
    print("output written to: " + str(Path(args.output, output_json)))
