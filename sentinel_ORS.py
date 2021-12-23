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
  
  #assets_hrefs=[
    #selected_item.assets["visual"].href,
    #selected_item.assets["B01"].href,
    #selected_item.assets["B02"].href,
    #selected_item.assets["B03"].href,
    #elected_item.assets["B04"].href,
    #selected_item.assets["B05"].href,
    #selected_item.assets["B06"].href,
    #selected_item.assets["B07"].href,
    #selected_item.assets["B08"].href,
    #selected_item.assets["B11"].href,
    #selected_item.assets["B12"].href
  #]
  #assest_hrefs=[selected_item.assets[key].href for key in selected_item.assets]
    #asset_hrefs[]=
  #asset_hrefs=least_cloudy_item.assets.values()
  
  #for i in range(len(least_cloudy_item.assets)):
    #asset_hrefs[i]=least_cloudy_item.assets.get(i).href
  #asset_hrefs=[banda.href for banda in least_cloudy_item.assets]
  
  asset_hrefs = [
    #least_cloudy_item.assets["visual"].href
    #find_asset_by_band_common_name(selected_item, "coastal").href,
    find_asset_by_band_common_name(selected_item, "red").href,
    find_asset_by_band_common_name(selected_item, "green").href,
    find_asset_by_band_common_name(selected_item, "blue").href,
    #find_asset_by_band_common_name(selected_item, "swir16").href,
    #find_asset_by_band_common_name(selected_item, "swir22").href,
    #find_asset_by_band_common_name(selected_item, "rededge").href,
    find_asset_by_band_common_name(selected_item, "nir").href
    #find_asset_by_band_common_name(selected_item, "rededge").href
  ]
  
  signed_hrefs = [pc.sign(asset_href) for asset_href in asset_hrefs]
  bands = [read_band(href) for href in signed_hrefs]
  multiband_data = np.stack(bands)
  
  file_location = "abfss://ors@dlsagrosaviaedcns.dfs.core.windows.net/bandas_10m"
  #X = np.swapaxes(np.swapaxes(multiband_data, 0, 2), 0, 1)
  #label=SGMM(X, S=3, Nc=5, P=2500, ERR=1)
  #bands.append(label)
  #multiband_data = np.stack(bands)
  
  for i in range(len(multiband_data)):
    banda = "banda" + str(i+1) +".tif"
    bandatemp = multiband_data[i]
    tiff = TIFF.open(banda, mode='w')
    tiff.write_image(bandatemp)
    tiff.close()
    dbutils.fs.cp("file:/databricks/driver/" + banda + "", file_location + selected_item.id + "_"+ banda) 

# COMMAND ----------

for j in range(len(items)):
  selected_item = sorted(items, key=lambda item: eo.ext(item).cloud_cover)[j]
  
  #assets_hrefs=[
    #selected_item.assets["visual"].href,
    #selected_item.assets["B01"].href,
    #selected_item.assets["B02"].href,
    #selected_item.assets["B03"].href,
    #elected_item.assets["B04"].href,
    #selected_item.assets["B05"].href,
    #selected_item.assets["B06"].href,
    #selected_item.assets["B07"].href,
    #selected_item.assets["B08"].href,
    #selected_item.assets["B11"].href,
    #selected_item.assets["B12"].href
  #]
  #assest_hrefs=[selected_item.assets[key].href for key in selected_item.assets]
    #asset_hrefs[]=
  #asset_hrefs=least_cloudy_item.assets.values()
  
  #for i in range(len(least_cloudy_item.assets)):
    #asset_hrefs[i]=least_cloudy_item.assets.get(i).href
  #asset_hrefs=[banda.href for banda in least_cloudy_item.assets]
  
  asset_hrefs = [
    #least_cloudy_item.assets["visual"].href
    #find_asset_by_band_common_name(selected_item, "coastal").href,
    #find_asset_by_band_common_name(selected_item, "red").href,
    #find_asset_by_band_common_name(selected_item, "green").href,
    #find_asset_by_band_common_name(selected_item, "blue").href,
    find_asset_by_band_common_name(selected_item, "swir16").href,
    find_asset_by_band_common_name(selected_item, "swir22").href,
    find_asset_by_band_common_name(selected_item, "rededge").href,
    #find_asset_by_band_common_name(selected_item, "nir").href
    #find_asset_by_band_common_name(selected_item, "rededge").href
  ]
  
  signed_hrefs = [pc.sign(asset_href) for asset_href in asset_hrefs]
  bands = [read_band(href) for href in signed_hrefs]
  multiband_data = np.stack(bands)
  
  file_location = "abfss://ors@dlsagrosaviaedcns.dfs.core.windows.net/bandas_20m/"
  #X = np.swapaxes(np.swapaxes(multiband_data, 0, 2), 0, 1)
  #label=SGMM(X, S=3, Nc=5, P=2500, ERR=1)
  #bands.append(label)
  #multiband_data = np.stack(bands)
  
  for i in range(len(multiband_data)):
    banda = "banda" + str(i+1) +".tif"
    bandatemp = multiband_data[i]
    tiff = TIFF.open(banda, mode='w')
    tiff.write_image(bandatemp)
    tiff.close()
    dbutils.fs.cp("file:/databricks/driver/" + banda + "", file_location + selected_item.id + "_"+ banda) 

# COMMAND ----------

for j in range(len(items)):
  selected_item = sorted(items, key=lambda item: eo.ext(item).cloud_cover)[j]
  
  #assets_hrefs=[
    #selected_item.assets["visual"].href,
    #selected_item.assets["B01"].href,
    #selected_item.assets["B02"].href,
    #selected_item.assets["B03"].href,
    #elected_item.assets["B04"].href,
    #selected_item.assets["B05"].href,
    #selected_item.assets["B06"].href,
    #selected_item.assets["B07"].href,
    #selected_item.assets["B08"].href,
    #selected_item.assets["B11"].href,
    #selected_item.assets["B12"].href
  #]
  #assest_hrefs=[selected_item.assets[key].href for key in selected_item.assets]
    #asset_hrefs[]=
  #asset_hrefs=least_cloudy_item.assets.values()
  
  #for i in range(len(least_cloudy_item.assets)):
    #asset_hrefs[i]=least_cloudy_item.assets.get(i).href
  #asset_hrefs=[banda.href for banda in least_cloudy_item.assets]
  
  asset_hrefs = [
    #least_cloudy_item.assets["visual"].href
    find_asset_by_band_common_name(selected_item, "coastal").href,
    #find_asset_by_band_common_name(selected_item, "red").href,
    #find_asset_by_band_common_name(selected_item, "green").href,
    #find_asset_by_band_common_name(selected_item, "blue").href,
    #find_asset_by_band_common_name(selected_item, "swir16").href,
    #find_asset_by_band_common_name(selected_item, "swir22").href,
    #find_asset_by_band_common_name(selected_item, "rededge").href,
    #find_asset_by_band_common_name(selected_item, "nir").href
    #find_asset_by_band_common_name(selected_item, "rededge").href
  ]
  
  signed_hrefs = [pc.sign(asset_href) for asset_href in asset_hrefs]
  bands = [read_band(href) for href in signed_hrefs]
  multiband_data = np.stack(bands)
  
  file_location = "abfss://ors@dlsagrosaviaedcns.dfs.core.windows.net/bandas_60m/"
  #X = np.swapaxes(np.swapaxes(multiband_data, 0, 2), 0, 1)
  #label=SGMM(X, S=3, Nc=5, P=2500, ERR=1)
  #bands.append(label)
  #multiband_data = np.stack(bands)
  
  for i in range(len(multiband_data)):
    banda = "banda" + str(i+1) +".tif"
    bandatemp = multiband_data[i]
    tiff = TIFF.open(banda, mode='w')
    tiff.write_image(bandatemp)
    tiff.close()
    dbutils.fs.cp("file:/databricks/driver/" + banda + "", file_location + selected_item.id + "_"+ banda) 

# COMMAND ----------



least_cloudy_item = sorted(items, key=lambda item: eo.ext(item).cloud_cover)[0]

print(
    f"Choosing {least_cloudy_item.id} from {least_cloudy_item.datetime.date()}"
    f" with {eo.ext(least_cloudy_item).cloud_cover}% cloud cover"
)

# COMMAND ----------

least_cloudy_item.assets.keys()

#type(least_cloudy_item.assets.keys())
#assest_hrefs=[item.href for item in selected_item.assets.items()]
#asset_hrefs

# COMMAND ----------

selected_item = sorted(items, key=lambda item: eo.ext(item).cloud_cover)[0]
find_asset_by_band_common_name(selected_item, "visual").href

# COMMAND ----------

asset_href = least_cloudy_item.assets["visual"].href

# COMMAND ----------

signed_href = pc.sign(asset_href)

# COMMAND ----------

signed_href

# COMMAND ----------

# MAGIC %%bash
# MAGIC pip install rasterio

# COMMAND ----------

import rasterio
from rasterio import windows
from rasterio import features
from rasterio import warp

import numpy as np
from PIL import Image

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
target_w = 1500
target_h = (int)(target_w / aspect)
img.resize((target_w, target_h), Image.BILINEAR)


# COMMAND ----------

import matplotlib.pyplot as plt
plt.imshow(img)

# COMMAND ----------

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
