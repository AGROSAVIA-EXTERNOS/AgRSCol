# Databricks notebook source
from pystac_client import Client
from pystac.extensions.eo import EOExtension as eo
import planetary_computer as pc

# COMMAND ----------

area_of_interest = {
    "type": "Polygon",
    "coordinates": [
        [
            [
              -74.20191764831543,
              4.696793484410148
            ],
            [
              -74.2129898071289,
              4.696023601786466
            ],
            [
              -74.21968460083008,
              4.68661385665204
            ],
            [
              -74.21350479125975,
              4.684304172562671
            ],
            [
              -74.20372009277344,
              4.65898269196767
            ],
            [
              -74.2020034790039,
              4.657870578971854
            ],
            [
              -74.19608116149902,
              4.66163464662711
            ],
            [
              -74.19831275939941,
              4.672156819644733
            ],
            [
              -74.19719696044922,
              4.691147658775067
            ],
            [
              -74.20148849487305,
              4.691917546778364
            ],
            [
              -74.20191764831543,
              4.696793484410148
            ]
        ]
    ],
}

# COMMAND ----------

time_of_interest = "1990-01-01/2021-12-31"

# COMMAND ----------

catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

search = catalog.search(
    collections=["landsat-8-c2-l2"],
    intersects=area_of_interest,
    datetime=time_of_interest,
    query={"eo:cloud_cover": {"lt": 100}},
)

# Check how many items were returned
items = list(search.get_items())
print(f"Returned {len(items)} Items")

# COMMAND ----------

selected_item = sorted(items, key=lambda item: eo.ext(item).cloud_cover)[0]

print(
    f"Choosing {selected_item.id} from {selected_item.datetime.date()}"
    + f" with {selected_item.properties['eo:cloud_cover']}% cloud cover"
)

# COMMAND ----------

def find_asset_by_band_common_name(item, common_name):
    for asset in item.assets.values():
        asset_bands = eo.ext(asset).bands
        if asset_bands and asset_bands[0].common_name == common_name:
            return asset
    raise KeyError(f"{common_name} band not found")

# COMMAND ----------

asset_hrefs = [
    find_asset_by_band_common_name(selected_item, "coastal").href,
    find_asset_by_band_common_name(selected_item, "red").href,
    find_asset_by_band_common_name(selected_item, "green").href,
    find_asset_by_band_common_name(selected_item, "blue").href,
    find_asset_by_band_common_name(selected_item, "swir16").href,
    find_asset_by_band_common_name(selected_item, "swir22").href,
    find_asset_by_band_common_name(selected_item, "lwir11").href
]

# COMMAND ----------

signed_hrefs = [pc.sign(asset_href) for asset_href in asset_hrefs]

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


bands = [read_band(href) for href in signed_hrefs]

# COMMAND ----------

bands

# COMMAND ----------

multiband_data = np.stack(bands)

# COMMAND ----------

multiband_data

# COMMAND ----------

signed_hrefs

# COMMAND ----------

# MAGIC %r
# MAGIC library(rgdal)
# MAGIC library(raster)
# MAGIC image <- stack()

# COMMAND ----------

secretos = dbutils.secrets.listScopes()

for secreto in secretos:
  print(secreto.name)

# COMMAND ----------

multiband_data.shape

# COMMAND ----------

banda2 = multiband_data[6]
banda2

# COMMAND ----------

spark.conf.set (
  "fs.azure.account.key.dlsagrosaviaedcns.dfs.core.windows.net",
  dbutils.secrets.get (scope = "key-vault-secrets", key = "StorageAccountAccesKey")
)

# COMMAND ----------

file_location = "abfss://testtiff@dlsagrosaviaedcns.dfs.core.windows.net/imagenes/"

# COMMAND ----------

from libtiff import TIFF
for i in range(len(multiband_data)):
  banda = "banda" + str(i+1) +".tif"
  bandatemp = multiband_data[i]
  tiff = TIFF.open(banda, mode='w')
  tiff.write_image(bandatemp)
  tiff.close()
  dbutils.fs.cp("file:/databricks/driver/" + banda + "", file_location + banda) 

# COMMAND ----------

import urllib.parse
import rioxarray

url = '/dbfs/mnt/agrosaviatest/imagenes/banda7.tif'
rioxarray.open_rasterio(url)
