import xarray as xr
import cartopy.crs as ccrs
import numpy as np
import sys
import json
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def longitudetocoords(x):
    return (((x - 180) % 360) + 180) % 360


def coordstolongitude(x):
    return ((x + 180) % 360) - 180


### Extemes event dataset

# Define the path to the extremes events data in Zarr format
path2extreme = "/Net/Groups/BGI/work_1/scratch/s3/deepextremes/v2/EventCube_ranked_pot0.01_ne0.1.zarr"
# Open the Zarr dataset
data = xr.open_zarr(path2extreme)

# Select the Africa region and time period.
dsc = data.roll(longitude=180 * 4, roll_coords=True)
extremes = dsc.assign_coords(longitude=coordstolongitude(dsc.longitude)).sel(
    longitude=slice(-20, 55),
    latitude=slice(38, -35),
    time=slice("2015-01-01", "2015-12-31"),
)

### Earthnet2023 data

path2earthnet = "/Net/Groups/BGI/scratch/crobin/PythonProjects/EarthNet/earthnet-models-pytorch/Data_analysis/coordinates.py"
# Open the Json dataset
f = open(path2earthnet)
earthnet2023 = json.load(f)

# Remove the non-extreme events
extremes["layer"] = extremes["layer"].where(extremes["layer"] != 16, 0)


color_list = [
    "#ffffff",
    "#ffffbc",
    "#ffc77f",
    "#ffff36",
    "#ffad6e",
    "#ffca00",
    "#e7925d",
    "#ff7100",
    "#bd784c",
    "#ff1700",
    "#935d3b",
    "#bd0000",
    "#69422a",
    "#640000",
    "#3f2819",
    "#0b0000",
]

labels = [
    "Non-extreme",  # 00000 0
    "Heat wave",  # 00001 1
    "PEI-30",  # 00010 2
    "Heat wave & PEI-30",  # 00011 3
    "PEI-90",  # 00100 4
    "PEI-90 & Heat wave",  # 00101 5
    "PEI-90 & PEI-30",  # 00110 6
    "PEI-90 & PEI-30 & Heat wave",  # 00111 7
    "PEI-180",  # 01000 8
    "PEI-180 & Heat wave",  # 01001 9
    "PEI-180 & PEI-30 ",  # 01010 10
    "PEI-180 & PEI-30 & Heat wave",  # 01011 11
    "PEI-180 & PEI-60",  # 01100 12
    "PEI-180 & PEI-60 & Heat wave",  # 01101 13
    "PEI-180 & PEI-60 & PEI-30",  # 01110 14
    "PEI-180 & PEI-60 & PEI-30 & Heat wave",  # 01111 15
]

# Define the color map using a list of color.
cmap = matplotlib.colors.ListedColormap(color_list)
# Generate a colormap index based on discrete intervals.
norm = matplotlib.colors.BoundaryNorm([i - 0.5 for i in range(len(labels) + 1)], cmap.N)

# Setup the initial plot
fig = plt.figure(figsize=(10, 5))
ax = plt.axes(
    projection=ccrs.PlateCarree(),
)  # the data's projection
p = extremes.isel(time=0).layer.plot.imshow(
    cmap=cmap,
    norm=norm,
    add_colorbar=False,
    transform=ccrs.PlateCarree(),  # the data's projection
    subplot_kws={"projection": ccrs.Mollweide(central_longitude=0)},
    animated=True,
)
# Set up static features - coastlines, political borders etc.
ax.coastlines()

# Define the colorbar
cbar = fig.colorbar(
    p,
    boundaries=[i for i in range(len(labels) + 1)],
    spacing="uniform",
)  # , fraction=0.03, boundaries=[i-.5 for i in range(len(labels))])
cbar.set_label("Extreme event intensity")
cbar.set_ticks(ticks=[i for i in range(len(labels))], labels=labels)


# Animation
def animation_function(t):
    ax.set_title(extremes.isel(time=t).time.dt.date.values)
    anim = p.set_array(np.squeeze(extremes.isel(time=t).to_array()))
    # plt.colorbar(p, orientation='vertical', shrink=0.7, label='Extreme event intensity')


animation = FuncAnimation(
    fig, animation_function, frames=len(extremes.time), interval=15
)

# Save the animation as a GIF
animation.save("animation_extremes.gif", writer=PillowWriter(fps=5))

plt.show()
