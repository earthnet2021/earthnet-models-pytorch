import xarray as xr
import cartopy.crs as ccrs
import numpy as np
import sys
from scipy.ndimage import binary_dilation
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def longitudetocoords(x):
    return (((x - 180) % 360) + 180) % 360


def coordstolongitude(x):
    return ((x + 180) % 360) - 180


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

labels_veg = [
    "10 - Tree cover",
    "20 - Shrubland",
    "30 - Grassland",
    "40 - Cropland",
    "50 - Built-up",
    "60 - Bare / sparse vegetation",
    "70 - Snow and Ice",
    "80 - Permanent water bodies",
    "90 - Herbaceous wetland",
    "95 - Mangroves",
    "100 - Moss and lichen",
]


### Extemes event dataset

# # Define the path to the extremes events data in Zarr format
# path2extreme = "/Net/Groups/BGI/work_1/scratch/s3/deepextremes/v2/EventCube_ranked_pot0.01_ne0.1.zarr"
# # Open the Zarr dataset
# data = xr.open_zarr(path2extreme).load()
#
# # Select the Africa region and time period.
# dsc = data.roll(longitude=180 * 4, roll_coords=True)
# extremes = dsc.assign_coords(longitude=coordstolongitude(dsc.longitude)).sel(
#     longitude=slice(-20, 55),
#     latitude=slice(38, -35),
#     time=slice("2015-01-01", "2015-3-31"),
# )
# Remove the non-extreme events
# extremes["layer"] = extremes["layer"].where(extremes["layer"] != 16, 0)

### Earthnet2023 data

path2earthnet = "/Net/Groups/BGI/scratch/crobin/PythonProjects/EarthNet/earthnet2023_veg_type_small2.zarr"
earthnet = (
    xr.open_zarr(path2earthnet)
    .load()
    .sel(lon=slice(-20, 55), lat=slice(38, -35), time=slice("2015-01-01", "2021-3-31"))
)

earthnet["veg_type"] = earthnet["veg_type"].where(earthnet["veg_type"] != 10, 1)
earthnet["veg_type"] = earthnet["veg_type"].where(earthnet["veg_type"] != 20, 2)
earthnet["veg_type"] = earthnet["veg_type"].where(earthnet["veg_type"] != 30, 3)
earthnet["veg_type"] = earthnet["veg_type"].where(earthnet["veg_type"] != 40, 4)
earthnet["veg_type"] = earthnet["veg_type"].where(earthnet["veg_type"] != 50, 5)
earthnet["veg_type"] = earthnet["veg_type"].where(earthnet["veg_type"] != 60, 6)
earthnet["veg_type"] = earthnet["veg_type"].where(earthnet["veg_type"] != 70, 7)
earthnet["veg_type"] = earthnet["veg_type"].where(earthnet["veg_type"] != 80, 8)
earthnet["veg_type"] = earthnet["veg_type"].where(earthnet["veg_type"] != 90, 9)
earthnet["veg_type"] = earthnet["veg_type"].where(earthnet["veg_type"] != 95, 10)
earthnet["veg_type"] = earthnet["veg_type"].where(earthnet["veg_type"] != 100, 11)

# Define the color map using a list of color.
cmap_extreme = matplotlib.colors.ListedColormap(color_list)
# Generate a colormap index based on discrete intervals.
norm_extreme = matplotlib.colors.BoundaryNorm(
    [i - 0.5 for i in range(len(labels) + 1)], cmap_extreme.N
)

# Setup the initial plot
fig = plt.figure(figsize=(10, 5))
ax = plt.axes(
    projection=ccrs.PlateCarree(),
)  # the data's projection
# p = extremes.isel(time=0).layer.plot.imshow(
#    cmap=cmap_extreme,
#    norm=norm_extreme,
#    add_colorbar=False,
#    transform=ccrs.PlateCarree(),  # the data's projection
#    subplot_kws={"projection": ccrs.Mollweide(central_longitude=0)},
#    animated=True,
# )
#
cmap_veg = plt.get_cmap("Set3", lut=11)
norm_veg = matplotlib.colors.BoundaryNorm(
    [i - 0.5 for i in range(len(labels_veg) + 1)], cmap_veg.N
)
cmap_veg = cmap_veg(np.arange(cmap_veg.N))
cmap_veg[0, -1] = 0
cmap_veg = matplotlib.colors.ListedColormap(cmap_veg)
# my_cmap = matplotlib.colors.ListedColormap(my_cmap)
e = earthnet.isel(time=0).veg_type.plot.imshow(
    cmap=cmap_veg,
    norm=norm_veg,
    transform=ccrs.PlateCarree(),  # the data's projection
    subplot_kws={"projection": ccrs.Mollweide(central_longitude=0)},
    animated=True,
)
# Set up static features - coastlines, political borders etc.
ax.coastlines()

# Define the colorbar
cbar = fig.colorbar(
    e,
    boundaries=[i for i in range(len(labels_veg) + 1)],
    spacing="uniform",
)  # , fraction=0.03, boundaries=[i-.5 for i in range(len(labels))])from matplotlib import pyplot as pltcbar.set_label("Extreme event intensity")
cbar.set_ticks(ticks=[i for i in range(len(labels_veg))], labels=labels_veg)

# Define the colorbar
# cbar = fig.colorbar(
#     p,
#     boundaries=[i for i in range(len(labels) + 1)],
#     spacing="uniform",
# )  # , fraction=0.03, boundaries=[i-.5 for i in range(len(labels))])from matplotlib import pyplot as plt

# cbar.set_label("Extreme event intensity")
# cbar.set_ticks(ticks=[i for i in range(len(labels))], labels=labels)

# Animation
def animation_function(t):
    ax.set_title(earthnet.isel(time=t).time.dt.date.values)
    # anim = p.set_array(np.squeeze(extremes.isel(time=t).to_array()))
    anim2 = e.set_array(np.squeeze(earthnet.isel(time=t).to_array()))

    # plt.colorbar(p, orientation='vertical', shrink=0.7, label='Extreme event intensity')

animation = FuncAnimation(
    fig, animation_function, frames=len(earthnet.time), interval=50
)

# Save the animation as a GIF
# animation.save("animation_extremes.gif", writer=PillowWriter(fps=5))

plt.show()
