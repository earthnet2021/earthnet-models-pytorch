
import argparse

import numpy as np
import pandas as pd
import xarray as xr
import dask.dataframe as dd

from pathlib import Path
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import warnings
warnings.filterwarnings('ignore')



def compute_metrics(targ, pred, name_ndvi_pred = "ndvi_pred", subset_hq = True):

    metrics = {}

    nir = targ.s2_B8A
    red = targ.s2_B04

    mask_full = ((targ.s2_dlmask < 1) & targ.s2_SCL.isin([1,2,4,5,6,7]))
    mask = mask_full.sel(time = pred.time)

    targ_ndvi_full = ((nir - red) / (nir + red + 1e-8)).where(mask_full == 1, np.NaN).clip(-1, 1)
    targ_ndvi = targ_ndvi_full.sel(time = pred.time)
    pred_ndvi = pred[name_ndvi_pred].clip(-1, 1)


    mse = ((targ_ndvi - pred_ndvi)**2).mean("time")

    targ_var = targ_ndvi.var("time")
    targ_std = targ_ndvi.std("time")

    


    metrics["nnse"] = 1 / (2 - (1 - (((targ_ndvi - pred_ndvi)**2).sum("time") / ((targ_ndvi - targ_ndvi.mean("time"))**2).sum("time"))))#1 / (2 - (1 - (mse / targ_var)))

    metrics["n_obs_full"] = (mask_full == 1).sum("time")
    metrics["n_obs"] = (mask == 1).sum("time")

    metrics["sigma_targ"] = targ_std
    metrics["sigma_pred"] = pred_ndvi.where(mask == 1, np.NaN).std("time")

    metrics["bias"] = targ_ndvi.mean("time") - pred_ndvi.where(mask == 1, np.NaN).mean("time")

    metrics["min_ndvi_targ"] = targ_ndvi_full.min("time")

    metrics["rmse"] = mse**0.5

    for i in [0, 5, 10, 15]:

        metrics[f"rmse_{i}_{i+5}"] = ((targ_ndvi - pred_ndvi)**2).isel(time = slice(i, i+5)).mean("time")**0.5

    metrics["r"] = xr.corr(targ_ndvi, pred_ndvi, dim = "time")

    metrics["landcover"] = targ.esawc_lc
    metrics["geom"] = targ.geom_cls
    metrics["cop_dem"] = targ.cop_dem

    df = xr.Dataset(metrics).to_dataframe().drop(columns="sentinel:product_id", errors = "ignore")

    if subset_hq:
        df = df[(df.landcover < 41) & (df.min_ndvi_targ > 0.0) & (df.n_obs >= 10) & ((df.n_obs_full - df.n_obs) >= 3) & (df.sigma_targ > 0.1)]

    return df


def score_from_args(args):

    targetfile, predfile, name_ndvi_pred = args

    targ = xr.open_dataset(targetfile)
    pred = xr.open_dataset(predfile)

    curr_df = compute_metrics(targ, pred, name_ndvi_pred=name_ndvi_pred)
    curr_df["id"] = targetfile.stem
    curr_df["season"] = targetfile.parent.stem

    return curr_df


def score_over_dataset(testset_dir, pred_dir, score_dir, name_ndvi_pred = "ndvi_pred", verbose = True, num_workers = 1):

    testset_dir = Path(testset_dir)
    pred_dir = Path(pred_dir)
    score_dir = Path(score_dir)
    score_dir.mkdir(exist_ok=True, parents=True)
    if verbose:
        print(f"scoring {testset_dir} against {pred_dir}")

    regions = sorted([x.stem for x in testset_dir.iterdir() if x.is_dir()])

    for region in tqdm(regions, desc = "Region", position = 0, leave = True) if verbose else regions:

        targetfiles = sorted(list((testset_dir/region).glob("*.nc")))

        predfiles = []
        inputargs = []
        for targetfile in targetfiles:
            cubename = targetfile.name
            predfile = pred_dir/region/cubename
            predfiles.append(predfile)
            inputargs.append([targetfile, predfile, name_ndvi_pred])
        
        

        with ProcessPoolExecutor(max_workers = num_workers) as pool:
            if verbose:
                dfs = list(tqdm(pool.map(score_from_args, inputargs), total = len(inputargs), desc = "Minicube", position=1, leave=False))
            else:
                dfs = list(pool.map(score_from_args, inputargs))

        df = pd.concat(dfs).reset_index()

        #df["season"] = region.split("_")[-1][:3]
        #df["year"] = 2000 + int(region.split("_")[-1][3:5])

        df.to_parquet(score_dir/f"scores_en21x_{region}.parquet", compression='snappy')

        del df, dfs

    return

def summarize_scores(score_dir, verbose = True):
    score_dir = Path(score_dir)
    df = dd.read_parquet(score_dir/"scores_en21x_*.parquet")
    nnse_per_landcover = (df.groupby(["landcover"])["nnse"].mean()).compute()
    rmse_per_landcover = (df.groupby(["landcover"])["rmse"].mean()).compute()
    R2_per_landcover = (df.groupby(["landcover"])["r"].apply(lambda x: (x**2).mean(), meta = ("R2", float))).compute()
    biasabs_per_landcover = (df.groupby(["landcover"])["bias"].apply(lambda x: x.abs().mean(), meta = ("biasabs", float))).compute()
    metrics = {
        "nse": 2- 1/nnse_per_landcover.mean(),
        "rmse": rmse_per_landcover.mean(),
        "R2": R2_per_landcover.mean(),
        "biasabs": biasabs_per_landcover.mean()
    }
    for landcover, lc_val in zip(["forest", "shrub", "grass", "crop"], [10, 20, 30, 40]):
        metrics[f"nse_{landcover}"] = 2- 1/nnse_per_landcover.loc[lc_val]
        metrics[f"rmse_{landcover}"] = rmse_per_landcover.loc[lc_val]
        metrics[f"R2_{landcover}"] = R2_per_landcover.loc[lc_val]
        metrics[f"biasabs_{landcover}"] = biasabs_per_landcover.loc[lc_val]
    pd.DataFrame.from_dict(metrics, orient = "index").to_csv(score_dir/"metrics_en21x.csv")
    if verbose:
        print(metrics)
    return metrics

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('testset_dir', type = str)
    parser.add_argument('pred_dir', type = str)
    parser.add_argument('score_dir', type = str)
    
    args = parser.parse_args()

    score_over_dataset(testset_dir=args.testset_dir, pred_dir=args.pred_dir, score_dir=args.score_dir, verbose = True, num_workers=20)
    summarize_scores(args.score_dir)