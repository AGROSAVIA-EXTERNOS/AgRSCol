# Databricks notebook source
from pystac_client import Client
from pystac.extensions.eo import EOExtension as eo
import planetary_computer as pc
from libtiff import TIFF

# COMMAND ----------

area_of_interest = {
    "type": "Polygon",
    "coordinates": [
        [
          [-73.7, 5.85], 
          [-73.3, 5.85], 
          [-73.3, 6.2], 
          [-73.7, 6.2], 
          [-73.7, 5.85]
        ]
    ],
}

# COMMAND ----------

time_of_interest = "2015-09-01/2016-02-28"

# COMMAND ----------

catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

search = catalog.search(
    collections=["sentinel-2-l2a"],
    intersects=area_of_interest,
    datetime=time_of_interest,
    query={"eo:cloud_cover": {"lt": 100}},
)

# Check how many items were returned
items = list(search.get_items())
print(f"Returned {len(items)} Items")

# COMMAND ----------

spark.conf.set (
  "fs.azure.account.key.dlsagrosaviaedcns.dfs.core.windows.net",
  dbutils.secrets.get (scope = "key-vault-secrets", key = "StorageAccountAccesKey")
)

# COMMAND ----------

def find_asset_by_band_common_name(item, common_name):
    for asset in item.assets.values():
        asset_bands = eo.ext(asset).bands
        if asset_bands and asset_bands[0].common_name == common_name:
            return asset
    raise KeyError(f"{common_name} band not found")

# COMMAND ----------

import rasterio
from rasterio import windows
from rasterio import features
from rasterio import warp

import numpy as np
from PIL import Image


def read_band(href):
    with rasterio.open(href) as ds:
        aoi_bounds = features.bounds(area_of_interest)
        warped_aoi_bounds = warp.transform_bounds("epsg:4326", ds.crs, *aoi_bounds)
        aoi_window = windows.from_bounds(transform=ds.transform, *warped_aoi_bounds)
        return ds.read(1, window=aoi_window)

# COMMAND ----------

least_cloudy_item = sorted(items, key=lambda item: eo.ext(item).cloud_cover)[3]

print(
    f"Choosing {least_cloudy_item.id} from {least_cloudy_item.datetime.date()}"
    f" with {eo.ext(least_cloudy_item).cloud_cover}% cloud cover"
)

# COMMAND ----------

selected_item = sorted(items, key=lambda item: eo.ext(item).cloud_cover)[0]
find_asset_by_band_common_name(selected_item, "coastal").href

# COMMAND ----------

asset_href = least_cloudy_item.assets["visual"].href
signed_href = pc.sign(asset_href)

# COMMAND ----------

signed_href

# COMMAND ----------

with rasterio.open(signed_href) as ds:
    aoi_bounds = features.bounds(area_of_interest)
    warped_aoi_bounds = warp.transform_bounds("epsg:4326", ds.crs, *aoi_bounds)
    aoi_window = windows.from_bounds(transform=ds.transform, *warped_aoi_bounds)
    band_data = ds.read(window=aoi_window)

# COMMAND ----------

img = Image.fromarray(np.transpose(band_data, axes=[1, 2, 0]))
w = img.size[0]
h = img.size[1]
aspect = w / h
target_w = 2000
target_h = (int)(target_w / aspect)
img.resize((target_w, target_h), Image.BILINEAR)

# COMMAND ----------

import matplotlib.pyplot as plt
plt.imshow(img)
