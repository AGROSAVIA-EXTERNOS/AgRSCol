# Databricks notebook source
# MAGIC %md
# MAGIC # Clasificación no supervisada de pixeles
# MAGIC El objetivo de esta guía es mostrar como se puede realizar clasificación no supervisada de pixeles. La clasificación de pixeles convencional utiliza información a nivel de pixel de la información espectral disponible en la imágen (satelital o aérea), desconociendo la relación de un pixel con sus vecinos. Un reciente artículo científico describe un algoritmo que explota la información espacial y espectral del vecindario de un pixel para obtener una clasificación. En esta guía se incluye una implementación realizada por nuestro equipo y muestra los resultados obtenidos.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Carga de librerías necesarias
# MAGIC A continuación se cargarán las librerías necesarias para poder acceder a listado de imágenes que contienen una región de interés. El listado de librerías es

# COMMAND ----------

from pystac_client import Client
from pystac.extensions.eo import EOExtension as eo
import planetary_computer as pc

# COMMAND ----------

# MAGIC %md
# MAGIC ## Definición de zona de consulta
# MAGIC Definimos la zona a buscar dentro del catalogo de lansat. En este caso consultaremos la zona de nuestro centro de investigación principal, Tibaitatá, en Mosquera Cundinamarca.

# COMMAND ----------

area_of_interest = {
    "type": "Polygon",
    "coordinates": [
        [
            [-74.20191764831543,4.696793484410148],
            [-74.2129898071289,4.696023601786466],
            [-74.21968460083008,4.68661385665204],
            [-74.21350479125975,4.684304172562671],
            [-74.20372009277344,4.65898269196767],
            [-74.2020034790039,4.657870578971854],
            [-74.19608116149902,4.66163464662711],
            [-74.19831275939941,4.672156819644733],
            [-74.19719696044922,4.691147658775067],
            [-74.20148849487305,4.691917546778364],
            [-74.20191764831543,4.696793484410148]
        ]
    ],
}

# COMMAND ----------

# MAGIC %md
# MAGIC Definimos un intervalo de fechas de toma de las imágenes satelitales, desde 1990 hasta el 2021.

# COMMAND ----------

time_of_interest = "1990-01-01/2021-12-31"

# COMMAND ----------

# MAGIC %md
# MAGIC y obtenemos la lista de imágenes satelitales multiespectrales disponibles mediante

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

# MAGIC %md
# MAGIC ## Seleccionar y visualizar una imagen de la lista
# MAGIC Seleccionaremos la imágen satelital con el mínimo de nubosidad. Esto se logra primero ordenando por porcentaje de nubosidad las imágenes disponibles y luego eligiendo el primer elemento.

# COMMAND ----------

selected_item = sorted(items, key=lambda item: eo.ext(item).cloud_cover)[0]

print(
    f"Choosing {selected_item.id} from {selected_item.datetime.date()}"
    + f" with {selected_item.properties['eo:cloud_cover']}% cloud cover"
)

# COMMAND ----------

# MAGIC %md
# MAGIC para este caso, tenemos un imágen con un mínimo de 5.13 por ciento de nubosidad.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualización de imágen seleccionada
# MAGIC Para ver la imágen seleccionada, debemos usar la información disponible en el espectro visible (rojo, verde y azul). Para ello definimos una función que nos permita buscar la información asociada a este espectro.

# COMMAND ----------

def find_asset_by_band_common_name(item, common_name):
    for asset in item.assets.values():
        asset_bands = eo.ext(asset).bands
        if asset_bands and asset_bands[0].common_name == common_name:
            return asset
    raise KeyError(f"{common_name} band not found")

# COMMAND ----------

# MAGIC %md
# MAGIC Usamos esta función para obtener cada una de las bandas del espctro visible.

# COMMAND ----------

asset_hrefs = [
    find_asset_by_band_common_name(selected_item, "red").href,
    find_asset_by_band_common_name(selected_item, "green").href,
    find_asset_by_band_common_name(selected_item, "blue").href,
]

# COMMAND ----------

# MAGIC %md
# MAGIC y creamos las direcciones de dichas imagenes usando nuestro login de planetary computer.

# COMMAND ----------

signed_hrefs = [pc.sign(asset_href) for asset_href in asset_hrefs]

# COMMAND ----------

# MAGIC %md
# MAGIC Para visualizar la imágen usamos la librería rasterio.

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
rescaled = multiband_data.astype(float)
min_value, max_value = rescaled.min(), rescaled.max()
rescaled = ((rescaled - min_value) * 255) / (max_value - min_value)
byte_data = rescaled.astype("ubyte")
img=Image.fromarray(np.transpose(byte_data, axes=[1, 2, 0]))

import matplotlib.pyplot as plt
plt.imshow(img)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Algoritmo de clasificación no supervisada
# MAGIC Las imágenes espectrales se diferencian de las imágenes convencionales porque incluyen una mayor cantidad de información del espectro, visible y no visible. La clasificación no supervisada de los pixeles de estas imágenes puede servir para sugerir grupos de pixeles dentro de la imagen con información espectral similar que puedan asociarse a eventos relevantes dentro del quehacer de la agricultura. Estas clases podrían asociarse a cultivos, enfermedades, suelos desnudos, problemas de irrigación, entre otros. Esta clasificación no supervisada constituye el primer paso para la construcción de sistemas automatizados de detección de eventos, permitiendo la asistencia del etiquetado de imágenes, teniendo en cuenta la capacidad original de los datos para discriminar grupos diferentes de pixeles. En otras palabras, permite medir el potencial de los datos para diferenciar entre eventos en un escena.
# MAGIC 
# MAGIC Dicha clasifica requiere de algoritmos especializados que permitan explotar la información espectral. Una posible implementación del algoritmo del estado del arte de imágenes espectrales sería la presentada a continuación

# COMMAND ----------

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

# MAGIC %md
# MAGIC Para usar el algoritmo, llamamos la función que lo empaqueta, preparando antes la imagen multiespectral con la primera línea del bloque a continuación

# COMMAND ----------

  X = np.swapaxes(np.swapaxes(multiband_data, 0, 2), 0, 1)
  label=SGMM(X, S=3, Nc=5, P=2500, ERR=1)
