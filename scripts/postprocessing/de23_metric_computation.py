from pathlib import Path
import xarray as xr
import numpy as np
import json
from tqdm import tqdm
import multiprocessing as mp
import os
from scipy.stats import pearsonr
from argparse import ArgumentParser


def calculate_variable_statistics(target, pred, anomalie=False, subset="all", cubename="", method = "by_frame", season = "DJF"):
    """Function to calculate statistics for each variable in a sample"""
    t0 = pred.time.values[0]
    target_len = len(pred.time)
    # time start index
    it = np.where(target.time==t0)[0][0]
    # first target_len steps, every 5 days
    tind = range(it,it + target_len, 1)
    # event_time start index
    ie = np.where(target.event_time == t0-np.timedelta64(12,'h'))[0][0]
    # every day
    eind = range(ie, ie + 5*target_len, 1)

    targ = (target.B8A - target.B04) / (target.B8A + target.B04 + 1e-8)[
        tind, ...
    ]
    if anomalie:
        targ = targ - target.msc[tind, ...]

    pred = pred.ndvi_pred

    # if any day in the test period is extreme, tag prediction as extreme (could be adapted by type of extreme)
    extremes = target.events[eind, ...]
    if np.logical_and(1 <= extremes, extremes <= 15).any():
        extreme = True
    else:
        extreme = False

    # set mask = 1 where data are OK
    cloud_mask = target.cloudmask_en[tind, ...] < 1.0
    lc = target.SCL[tind, ...]
    lc_mask = lc == 4 # lc != 4 #(lc <= 40) | (lc >= 90)
    # lc_mask = np.expand_dims(lc_mask, axis=0)
    # lc_mask = np.repeat(lc_mask, target_len, axis=0)

    mask = cloud_mask * lc_mask

    targ = np.where(targ * mask == 0, np.nan, targ)
    pred = np.where(pred * mask == 0, np.nan, pred)
    n_obs = np.count_nonzero(~np.isnan(targ), axis=0)  # per pixel in space

    if np.sum(n_obs) / (128 * 128 * target_len) < 0.05:
        # not enough data
        return cubename, season, extreme, None, None, None, round(np.sum(n_obs) / (128 * 128 * target_len)*100, ndigits=3)
    elif subset == "extreme" and not extreme:
        # extreme subset should not have non extreme obs.
        return cubename, season, extreme, None, None, None, None
    elif subset == "non_extreme" and extreme:
        # non extreme subset should not have extreme obs.
        return cubename, season, extreme, None, None, None, None
    #else:

    if method == "by_frame":
        # stats are first computed over time and then averaged over space by minicube
        sum_squared_error = np.nansum((targ - pred) ** 2, axis=0)
        rmse = sum_squared_error / (n_obs + 1e-8)
        r2 = np.zeros((128, 128))
        # cov = np.zeros((128, 128))
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

                    # cov[i, j] = np.sum(dev_targ * dev_targ) / (
                    #     np.sqrt(np.sum(dev_targ_squared))
                    #     * np.sqrt(np.sum(dev_targ_squared))
                    #     + 1e-8
                    # )
                    sum_squared_dev[i,j] = np.sum(dev_targ_squared)
                else:
                    sum_squared_dev[i, j] = np.nan
                    r2[i, j] = np.nan

                    # if len(x[~nas]) > 1 and ~np.all(x[~nas] == x[~nas][0]):
                    #     r2[i, j] = (pearsonr(x[~nas], y[~nas]).statistic) ** 2
                    #     cov[i, j] = (pearsonr(x[~nas], x[~nas]).statistic) ** 2
                    # else:
                    #     if np.count_nonzero(~np.isnan(x)) < 2:
                    #         cov[i, j] = np.nan
                    #         r2[i, j] = np.nan
                    #     else:
                    #         cov[i, j] = 1
                    #         if x[~nas].all() == y[~nas].all():
                    #             r2[i, j] = 1
                    #         else:
                    #             r2[i, j] = np.nan

        # nse = 1 - (sum_squared_error / (cov**2 + 1e-8))
        nse = 1 - (sum_squared_error / (sum_squared_dev + 1e-8))

        if np.count_nonzero(~np.isnan(rmse)) == 0:
            rmse = None
        else:
            rmse = np.nanmean(rmse)
        if np.count_nonzero(~np.isnan(nse)) == 0:
            nse = None
        else:
            nse = np.nanmean(nse)
        if np.count_nonzero(~np.isnan(r2)) == 0:
            r2 = None
        else:
            r2 = np.nanmean(r2)
        return cubename, season, extreme, rmse, nse, r2, round(np.sum(n_obs) / (128 * 128 * target_len)*100, ndigits=3)
    elif method == "overall":
        sum_squared_error = np.nansum((targ - pred) ** 2)
        rmse = sum_squared_error / (np.sum(n_obs) + 1e-8)
        nas = np.logical_or(np.isnan(targ), np.isnan(pred))
        r2 = (pearsonr(targ[~nas], pred[~nas]).statistic) ** 2
        sum_squared_dev = np.nansum((targ - np.nanmean(targ)) ** 2)
        nse = 1 - (sum_squared_error / (sum_squared_dev + 1e-8))
        return cubename, season, extreme, rmse, nse, r2, round(np.sum(n_obs) / (128 * 128 * target_len)*100, ndigits=3)
    else:
        raise NameError("Method " + method +" is not defined.") 

def get_season(date:np.datetime64) -> str:
    # 1: MAM, 2: JJA, 3: SON, 4: DJF
    seasons = ['MAM', 'JJA', 'SON', 'DJF',]
    season = int(str(np.datetime64(date, 'M'))[5:]) // 3 -1
    return seasons[3 if season < 0 else season]


def get_name(path: Path) -> str:
        """Helper function gets Cubename from a Path

        Args:
            path (Path): One of Path/to/cubename.npz and Path/to/experiment_cubename.npz

        Returns:
            [str]: cubename (has format mc_lon_lat)
        """
        components = path.name.split("/")
        cubename = components[-1][0:-16]
        season = get_season(np.datetime64(components[-1][-15:-5]))
        return cubename, season


def calculate_sample_statistics(pred_path, anomalie=False, method="by_frame"):
    cubename, season = get_name(pred_path)
    try :
        pred = xr.open_dataset(pred_path, engine="zarr").load()
        # get test paths
        test_path = list(Path("/Net/Groups/BGI/tscratch/mweynants/dx-minicubes").glob("full/*/" + cubename + "*.zarr"))[0]
        target = xr.open_dataset(test_path, engine="zarr").load()

        if anomalie:
            pred_anomalie_path = list(
                Path(
                    "/Net/Groups/BGI/scratch/crobin/PythonProjects/EarthNet/earthnet-models-pytorch/experiments/en23/convlstm_ae/convlstm_ae/config_16.10.2023_anomalie_NDVI/preds/iid"
                ).glob("*/" + name)
                )[0]
            pred_anomalie = xr.open_dataset(pred_anomalie_path, engine="zarr").load()
            return [
                calculate_variable_statistics(target, prediction, anomalie, "all", cubename, method, season,)
                for prediction, anomalie in [(pred, False), (pred_anomalie, True)]
            ]
        else:
            pred_anomalie = None
            return [
                calculate_variable_statistics(target, prediction, anomalie, "all", cubename, method, )
                for prediction, anomalie in [(pred, False)]
            ]
    except :
        return [(cubename, season, None, None, None, None, None)]

def calculate(args):
    result = calculate_sample_statistics(*args)
    return result

def process_samples_in_parallel(paths, anomalie=False, method = "by_frame"):
    "Function to process samples using parallel processing"
    num_cores = mp.cpu_count()
    print("numbers of cores availables: ", num_cores)
    pool = mp.Pool(processes=100)
    tasks = [(path, anomalie, method) for path in paths]
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
        help="metrics computation method. 'by_frame' or 'overall'"
    )

    args = parser.parse_args()

    bpath  = args.bpath
    components = bpath.split("/")
    output_json = "results_" + components[-4] + "_" + components[-2] + "_" + components[-1] + "_" + args.method + ".json"
    print(Path(args.output, output_json))
    
    basepath = Path(
        bpath, "preds/iid/"
    )

    # paths to test predictions
    paths = list(basepath.glob("*.zarr"))

    print("len of the dataset: ", len(paths))

    # Call the function to process samples in parallel
    sample_statistics = process_samples_in_parallel(paths, method = args.method)

    # Transpose the results to get statistics for each variable
    variable_statistics = list(map(list, zip(*sample_statistics)))

    # save the statistics for each variable
    data = {}
    for var_idx, var_stats in enumerate(variable_statistics):
        # var_name = ["absolute_ndvi", "anomalie_ndvi"][var_idx]
        cubename, season, extreme, rmse, nse, r2, valid_pixels = zip(*var_stats)
        rmse = np.float64(rmse).tolist()
        nse = np.float64(nse).tolist()
        r2 = np.float64(r2).tolist()
        # data[str(var_name)] = {
        data = {
            "cubename": [i for i in cubename],
            "season": [i for i in season],
            "extreme": [i for i in extreme],
            "rmse": [i for i in rmse], # [i for i in rmse if i is not (None or np.nan)],
            "nse": [i for i in nse], # [i for i in nse if i is not (None or np.nan)],
            "r2": [i for i in r2], # [i for i in r2 if i is not (None or np.nan)],
            "valid_pixels": [i for i in valid_pixels]
        }

    with open(Path(args.output, output_json), "w") as fp:
        json.dump(data, fp)

    # print(data)
    print("output written to: " + str(Path(args.output, output_json)))

