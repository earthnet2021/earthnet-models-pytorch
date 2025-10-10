import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import time

# Read your CSV file
df = pd.read_csv(
    "/Net/Groups/BGI/work_2/scratch/DeepExtremes/mc_earthnet_biome.csv", delimiter=","
)

# Function to get biome from a row
# Load the WWF biome dataset
biome_data_path = "/Net/Groups/BGI/scratch/crobin/PythonProjects/EarthNet/terr-ecoregions-TNC/tnc_terr_ecoregions.shp"
biome_data = gpd.read_file(biome_data_path)


def get_biome(row):
    point = Point(row["lon"], row["lat"])

    for _, row in biome_data.iterrows():
        if point.within(row["geometry"]):
            return row["WWF_MHTNAM"]

    return "Unknown"


# Load a world map dataset (included in geopandas)
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
africa = world[world["continent"] == "Africa"]


def is_in_africa(row, africa):
    # Create a Shapely Point from latitude and longitude
    point = Point(row["lon"], row["lat"])

    # Check if the point is in Africa
    return africa.contains(point).any()


# Apply the is_in_africa function to each row
df["in_africa"] = df.apply(lambda row: is_in_africa(row, africa), axis=1)

# Apply the function to create a new 'biome' column
# df["biome"] = df.apply(get_biome, axis=1)
df.to_csv("/Net/Groups/BGI/work_2/scratch/DeepExtremes/mc_earthnet_biome.csv")
