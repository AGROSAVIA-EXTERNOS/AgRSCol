# Databricks notebook source
# MAGIC %md
# MAGIC # Introducción a Planetary Computer 
# MAGIC Este cuaderno de Databricks busca ser una introducción a los comandos básicos y el acceso a la información satelital de los sensores disponibles en la recientemente lanzada plataforma <a href=https://planetarycomputer.microsoft.com/>Planetary Computer </a>. El objetivo es que las entidades de nuestro gobierno colombiano puedan empezar a usar esta herramienta usando servicios de la nube, bien sea contratados por la misma entidad o a través de iniciativas como el <a href=https://sandbox.datos.gov.co/#!/inicio>Data Sandbox de MinTic </a>.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cargar librerías de Planetary Computer.
# MAGIC Para acceder a la interfaz de la iniciativa de Microsoft Planetary Computer, es necesario cargar la librería correspondiente en Python. Antes de cargarla, es importante instalarla. La instalación en databricks se puede realizar de manera temporal mediante
# MAGIC ```
# MAGIC %%bash
# MAGIC pip install pystac-client
# MAGIC ```
# MAGIC La línea `%%bash` permite ejecutar en una celda comandos de terminal. `pip install pystac-client` permitirá instalar las librerías necesarias en python para poder acceder a las capacidades disponibles en Planetary Computer. Una vez instalada la librería, podemos accederla a través de python ejecutando

# COMMAND ----------

from pystac_client import Client

# COMMAND ----------

# MAGIC %md
# MAGIC Para listar las bases de datos disponibles en Planetary Computer, podemos usar los comandos

# COMMAND ----------

catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
collections = catalog.get_children()
for collection in collections:
    print(f"{collection.id} - {collection.title}")

# COMMAND ----------

# MAGIC %md
# MAGIC Dentro del catalogo disponible, notemos que tenemos acceso a datos del sensor Sentinel (sentinel-2-l2a - Sentinel-2 Level-2A). Revisemos el sensor disponible

# COMMAND ----------

sentinel = catalog.get_child('sentinel-2-l2a')
for band in sentinel.extra_fields['summaries']['eo:bands']:
    name = band['name']
    description = band['description']
    common_name = "" if 'common_name' not in band else f"({band['common_name']})"
    print(f"{name} {common_name}: {description} ")

# COMMAND ----------

# MAGIC %md
# MAGIC El sensor de sentinel tiene disponible 12 componentes de información. Revisemos el sensor de lansat

# COMMAND ----------

lansat = catalog.get_child('landsat-8-c2-l2')
for band in lansat.extra_fields['summaries']['eo:bands']:
    name = band['name']
    description = band['description']
    common_name = "" if 'common_name' not in band else f"({band['common_name']})"
    print(f"{name} {common_name}: {description} ")

# COMMAND ----------

# MAGIC %md
# MAGIC Tenemos 15 componentes de información disponibles en este catálogo.
