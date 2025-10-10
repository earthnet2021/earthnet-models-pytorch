import numpy as np
from pathlib import Path

# Define a function to compute the scale factor and add offset for data conversion
def compute_scale_and_offset(savepath, v, da, n=16):
    """
    From earthnet-minicuber. Based on Krios101's code above.
    Calculate the offset and scale factor for integer conversion of data.

    Parameters:
    - savepath: Path to save the data (not used in the computation).
    - v: Variable name (not used in the computation).
    - da: Data array to be converted.
    - n: Number of bits for integer representation (default is 16).

    Returns:
    - scale_factor: Scale factor for data conversion.
    - add_offset: Add offset for data conversion.
    """

    # Check if there are no NaN values in the data
    if ~np.isnan(da).all():
        vmin = np.nanmin(da).item()
        vmax = np.nanmax(da).item()
    else:
        vmin = np.nan
        vmax = np.nan

    # Calculate the scale factor to fit data in the available packed range
    scale_factor = (vmax - vmin) / (2**n - 1)

    # Translate the data range to be symmetric about zero
    add_offset = vmin + 2 ** (n - 1) * scale_factor

    return scale_factor, add_offset

# Define a function to save a NetCDF dataset
@staticmethod
def save_minicube_netcdf(minicube, savepath):
    """
    Save a given minicube dataset as a NetCDF file.

    Parameters:
    - minicube: Dataset to be saved.
    - savepath: Path where the NetCDF file will be saved.

    This function processes variables in the dataset and determines encoding options for saving.
    """

    # Convert 'savepath' to a Path object
    savepath = Path(savepath)

    # Initialize encoding dictionary
    encoding = {}

    # Iterate through variables in the dataset
    for v in list(minicube.variables):
        # Skip coordinates
        if v in ["time", "time_clim", "lat", "lon"]:
            continue

        # Check if interpolation_type attribute is 'linear'
        elif ("interpolation_type" in minicube[v].attrs) and (
            minicube[v].attrs["interpolation_type"] == "linear"
        ):
            scale_factor, add_offset = compute_scale_and_offset(
                savepath, v, minicube[v].values
            )
        else:
            scale_factor, add_offset = 1.0, 0.0

        # Define encoding options for the variable
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

    # Check if the file already exists and delete it if it does
    if savepath.is_file():
        savepath.unlink()
    else:
        savepath.parents[0].mkdir(exist_ok=True, parents=True)

    # Save the minicube dataset as a NetCDF file with specified encoding
    minicube.to_netcdf(savepath, encoding=encoding, compute=True)