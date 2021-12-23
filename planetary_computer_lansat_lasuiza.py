# Databricks notebook source
from pystac_client import Client
from pystac.extensions.eo import EOExtension as eo
import planetary_computer as pc
from libtiff import TIFF


# COMMAND ----------

# MAGIC %%sh
# MAGIC pip install pystac-client

# COMMAND ----------

area_of_interest = {
        "type": "Polygon",
        "coordinates": [
          [
            [
              -73.17896604537962,
              7.376235303681712
            ],
            [
              -73.18473815917969,
              7.373490162384474
            ],
            [
              -73.18164825439453,
              7.3640842003834095
            ],
            [
              -73.17525386810303,
              7.365786651743531
            ],
            [
              -73.17667007446288,
              7.371787740625952
            ],
            [
              -73.17761421203613,
              7.3745541726595825
            ],
            [
              -73.17896604537962,
              7.376235303681712
            ]
          ]
        ]
      }

# COMMAND ----------

time_of_interest = "1972-01-01/2021-12-31"

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

import matplotlib.pyplot as plt

def cubes_block(data, S):

    N = S * S      # window pixels number

    Nx, Ny, D = data.shape    # dimensions
    M         = Nx * Ny       # windows number

    data_padd = np.pad(data, [(int((S-1)/2),), (int((S-1)/2),), (0,)], mode='symmetric')    # data padding
    block     = np.zeros((N, M, D))     # block array init

    m = 0   # window index

    # block filling
    for i in range(Nx):
        for j in range(Ny):

            wind           = data_padd[i : i + S, j : j + S, :].copy()
            block[:, m, :] = wind.reshape(N, D)

            m += 1

    return block

def SGMM(X, S, Nc, P, ERR):
  # S               window size
  N     = S * S   # pixels number
  # Nc              clusters number
  # P               iterations number
  # ERR             minimum error

  # odd size checking
  if not S % 2:
      S = S + 1

  Nx, Ny, D  = X.shape     # row length, column lenght, bands number
  M          = Nx * Ny     # samples number

  w     = np.zeros((Nc, M))           # weights (3 per pixel)
  mu    = np.zeros((Nc, D))           # means (1 per cluster)
  sigma = np.zeros((Nc, D, D))        # covariance arrays (1 per cluster)
  gamma = np.random.rand(Nc, M, N)    # latent variable likelihood cubes block

  cube_block = cubes_block(X, S)
  cube_block = np.swapaxes(cube_block, 0, 1)     # data cubes block (M, N, D)

  for p in range(P):

      w_old     = w.copy()
      mu_old    = mu.copy()
      sigma_old = sigma.copy()

      gamma = gamma / np.sum(gamma, 0)[np.newaxis, :]  # normalization of priori

      # weights stimation
      g = np.sum(gamma, 2)
      w = g / N

      gk_sum = np.sum(g, 1)

      # mean stimation
      mu = np.sum((cube_block[np.newaxis]) * (gamma[:, :, :, np.newaxis]), (1, 2)) / (gk_sum[:, np.newaxis])

      # covariance stimation
      t     = cube_block[np.newaxis] - mu[:, np.newaxis, np.newaxis, :]
      sigma = np.sum(np.matmul(np.swapaxes((gamma[:, :, :, np.newaxis]) * t, 2, 3), t), 1) / \
              gk_sum[:, np.newaxis, np.newaxis]

      # latent variable stimation
      gamma = ((w) / ((np.sqrt(np.linalg.det(sigma)))[:, np.newaxis]))[:, :, np.newaxis] * \
              np.exp(-np.sum(t * np.swapaxes(np.matmul(np.linalg.inv(sigma)[:, np.newaxis], np.swapaxes(t, 2, 3)), 2, 3), 3) / 2)

      err = np.linalg.norm(w - w_old) + np.linalg.norm(mu - mu_old) + np.linalg.norm(sigma - sigma_old)

      if p > 1 and err < ERR:
          break

  label = np.argmax(w, 0)
  plt.imshow(label.reshape((Nx, Ny)) + 1, cmap='jet')
  plt.show()
  return label.reshape((Nx, Ny)) + 1

# COMMAND ----------

for j in range(len(items)):
  selected_item = sorted(items, key=lambda item: eo.ext(item).cloud_cover)[j]
  asset_hrefs = [
    find_asset_by_band_common_name(selected_item, "coastal").href,
    find_asset_by_band_common_name(selected_item, "red").href,
    find_asset_by_band_common_name(selected_item, "green").href,
    find_asset_by_band_common_name(selected_item, "blue").href,
    find_asset_by_band_common_name(selected_item, "swir16").href,
    find_asset_by_band_common_name(selected_item, "swir22").href,
    find_asset_by_band_common_name(selected_item, "lwir11").href
  ]
  
  signed_hrefs = [pc.sign(asset_href) for asset_href in asset_hrefs]
  bands = [read_band(href) for href in signed_hrefs]
  multiband_data = np.stack(bands)
  file_location = "abfss://testtiff@dlsagrosaviaedcns.dfs.core.windows.net/lansat/lasuiza/"
  
  X = np.swapaxes(np.swapaxes(multiband_data, 0, 2), 0, 1)
  label=SGMM(X, S=3, Nc=5, P=2500, ERR=1)
  bands.append(label)
  multiband_data = np.stack(bands)
  
  
  for i in range(len(multiband_data)):
    banda = "banda" + str(i+1) +".tif"
    bandatemp = multiband_data[i]
    tiff = TIFF.open(banda, mode='w')
    tiff.write_image(bandatemp)
    tiff.close()
    dbutils.fs.cp("file:/databricks/driver/" + banda + "", file_location + selected_item.id + "_"+ banda) 

# COMMAND ----------

for i in range(len(multiband_data)):
    banda = "banda" + str(i+1) +".tif"
    bandatemp = multiband_data[i]
    tiff = TIFF.open(banda, mode='w')
    tiff.write_image(bandatemp)
    tiff.close()
    dbutils.fs.cp("file:/databricks/driver/" + banda + "", file_location + selected_item.id + "_"+ banda) 

# COMMAND ----------

X.shape
bands.append(label.reshape((143, 87)) + 1)
i

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
multiband_data = np.stack(bands)

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
