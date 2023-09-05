from pathlib import Path
import datetime
import xarray as xr
import numpy as np
from tqdm import tqdm
import random
import multiprocessing as mp
from scipy.interpolate import interp1d
import os
import sys


def compute_scale_and_offset(savepath, v, da, n=16):
    """From earthnet-minicuber. Calculate offset and scale factor for int conversion.

    Based on Krios101's code above.
    """
    if ~np.isnan(da).all():
        vmin = np.nanmin(da).item()
        vmax = np.nanmax(da).item()
    else:
        vmin = np.nan
        vmax = np.nan

    # stretch/compress data to the available packed range
    scale_factor = (vmax - vmin) / (2**n - 1)

    # translate the range to be symmetric about zero
    add_offset = vmin + 2 ** (n - 1) * scale_factor

    return scale_factor, add_offset


@staticmethod
def save_minicube_netcdf(minicube, savepath):
    """From earthnet-minicuber."""
    savepath = Path(savepath)
    encoding = {}
    for v in list(minicube.variables):
        if v in ["time", "time_clim", "lat", "lon"]:
            continue

        elif ("interpolation_type" in minicube[v].attrs) and (
            minicube[v].attrs["interpolation_type"] == "linear"
        ):
            scale_factor, add_offset = compute_scale_and_offset(
                savepath, v, minicube[v].values
            )
        else:
            scale_factor, add_offset = 1.0, 0.0

        if (
            abs(scale_factor) < 1e-8
            or np.isnan(scale_factor)
            or (scale_factor == 1.0 and minicube[v].max() > 32766)
        ):
            encoding[v] = {"zlib": True, "complevel": 9}

        else:
            encoding[v] = {
                "dtype": "int16",
                "scale_factor": scale_factor,
                "add_offset": add_offset,
                "_FillValue": -32767,
                "zlib": True,
                "complevel": 9,
            }

    if savepath.is_file():
        savepath.unlink()
    else:
        savepath.parents[0].mkdir(exist_ok=True, parents=True)

    minicube.to_netcdf(savepath, encoding=encoding, compute=True)
