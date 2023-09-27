import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib as mpl
import cartopy.crs as ccrs
import datetime
import numpy as np
import sys

def longitudetocoords(x):
    return (((x - 180) % 360) + 180) % 360


def coordstolongitude(x):
    return ((x + 180) % 360) - 180


# Define the path to the extremes events data in Zarr format
path2cube = "/Net/Groups/BGI/work_1/scratch/s3/deepextremes/v2/EventCube_ranked_pot0.01_ne0.1.zarr"

# Open the Zarr dataset
data = xr.open_zarr(path2cube)

dsc = data.roll(longitude=180 * 4, roll_coords=True)
extremes = dsc.assign_coords(longitude=coordstolongitude(dsc.longitude)).sel(
    longitude=slice(-20, 55),
    latitude=slice(38, -35),
    time=slice("2015-01-01", "2021-12-31"),
)

extremes["layer"] = extremes["layer"].where(extremes["layer"] != 16, 0)
# extremes.layer.loc[extremes.layer.values == 16] = 0
# extremes[extremes.layer.values == 16].values = 0


cols = [
    "#ffffff",  # 00000000 # 0x00 # 0
    "#0000c8",  # 00000010 # 0x02 # 2
    "#000096",  # 00000100 # 0x04 # 4
    "#000096",  # 00000110 # 0x06 # 6
    "#000032",  # 00001000 # 0x08 # 8
    "#000032",  # 00001010 # 0x0a # 10
    "#000032",  # 00001100 # 0x0c # 12
    "#000032",  # 00001110 # 0x0e # 14
    "#afafaf",  # 00010000 # 0x10 # 16
]

labels = ['Label1', 'Label2', 'Label3', 'Label4', 'Label5','Label6', 'Label7']

# cmp = mpl.colors.ListedColormap(cols)
cmap = plt.cm.get_cmap("hot_r", 7)
# cbar.ax.set_xticklabels(['Low', 'Medium', 'High'])
map_proj = ccrs.Mollweide(central_longitude=0)

# Setup the initial plot
fig = plt.figure(figsize=(7, 5))
ax = plt.axes(
    projection=ccrs.PlateCarree(),
)  # the data's projection

p = extremes.isel(time=0).layer.plot.imshow(
    cmap=cmap,
    projection=ccrs.PlateCarree(),
    subplot_kws={"projection": map_proj},
    animated=True,
)

cbar = fig.colorbar(p,fraction=0.03,boundaries=[i-.5 for i in range(11)])
cbar.set_label('"Extreme event intensity"')
cbar.set_ticks(ticks=range(7),labels=labels)

# Set up static features - coastlines, political borders etc.
ax.coastlines()


def animation_function(t):
    ax.set_title(extremes.isel(time=t).time.dt.date.values)
    p.set_array(np.squeeze(extremes.isel(time=t).to_array()))
    # plt.colorbar(p, orientation='vertical', shrink=0.7, label='Extreme event intensity')


animation = FuncAnimation(
    fig, animation_function, frames=len(extremes.time), interval=15
)


# Save the animation as a GIF
# animation.save('animation_extremes.gif', writer=PillowWriter(fps=10))

plt.show()

