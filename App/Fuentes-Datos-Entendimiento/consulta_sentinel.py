# Databricks notebook source
# MAGIC %md
# MAGIC # Consulta catalogo Sentinel
# MAGIC Planetary Computer tiene la capacidad de buscar una determinada región dentro de las imágenes satelitales disponibles y retornar dicha región recortada. El siguiente ejemplo fue adaptado del <a href=https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a#Example-Notebook>ejemplo </a> disponible en la página de la herramienta. Aquí obtendremos la lista de de imágenes que contienen nuestra región de consulta en un intervalo de fechas determinadas.

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
# MAGIC Definimos la zona a buscar dentro del catalogo de sentinel. En este caso consultaremos la zona de nuestro centro de investigación principal, Tibaitatá, en Mosquera Cundinamarca.

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
    collections=["sentinel-2-l2a"],
    intersects=area_of_interest,
    datetime=time_of_interest,
    query={"eo:cloud_cover": {"lt": 100}},
)

# Check how many items were returned
items = list(search.get_items())
print(f"Returned {len(items)} Items")

# COMMAND ----------


