# Databricks notebook source
# MAGIC %python
# MAGIC # Azure Storage Account Name
# MAGIC storage_account_name = "dlsagrosaviaedcns"
# MAGIC 
# MAGIC # Azure Storage Account Key
# MAGIC storage_account_key = "2dvpXLfGMCe4L9EJUz0UXAq9eyw2qbOsyG7XEz63HGWJaSW9FCVI8TljA4QB0zU1Ac+oiPIlLVWv17vzeUM8aA=="
# MAGIC 
# MAGIC # Azure Storage Account Source Container
# MAGIC container = "imgtest"
# MAGIC 
# MAGIC # Set the configuration details to read/write
# MAGIC spark.conf.set("fs.azure.account.key.{0}.blob.core.windows.net".format(storage_account_name), storage_account_key)

# COMMAND ----------

# MAGIC %python
# MAGIC   dbutils.fs.mount(
# MAGIC    source = "wasbs://{0}@{1}.blob.core.windows.net".format(container, storage_account_name),
# MAGIC    mount_point = "/mnt/agrosavia",
# MAGIC    extra_configs = {"fs.azure.account.key.{0}.blob.core.windows.net".format(storage_account_name): storage_account_key}
# MAGIC   )
# MAGIC #/mnt/agrosavia  *agrosavia = este nombre lo pueden cambiar 

# COMMAND ----------

#Mostrar objetos del Punto de Montaje
#/mnt/agrosavia  *agrosavia = este nombre lo pueden cambiar 
display(dbutils.fs.ls("/mnt/agrosavia"))

# COMMAND ----------

# MAGIC %r
# MAGIC library(rgdal)
# MAGIC library(raster)

# COMMAND ----------

# MAGIC %r
# MAGIC dir('/dbfs/mnt/agrosavia/')

# COMMAND ----------

# MAGIC %r
# MAGIC #Cargar todos los nombres de las imágenes de bandas en una lista
# MAGIC list <- list.files(path='/dbfs/mnt/agrosavia/', full.names=TRUE)
# MAGIC #Creamos un stack con esas imágenes
# MAGIC image <- stack(list)
# MAGIC #Miremos las bandas que usamos
# MAGIC nlayers(image)
# MAGIC #Si hacemos image[], tendremos una matriz con filas como pixeles y columnas como espectro
# MAGIC # Hacemos kmeans a esa lista de pixeles
# MAGIC kMeansResult <- kmeans(image[], centers=6)
# MAGIC 
# MAGIC # ponemos la información en una capa y mostramos
# MAGIC result <- raster(image[[1]])
# MAGIC result <- setValues(result, kMeansResult$cluster)
# MAGIC plot(result)

# COMMAND ----------

# MAGIC %r
# MAGIC install.packages("RGISTools")

# COMMAND ----------

# MAGIC %sh
# MAGIC sudo apt-get install libfontconfig1-dev

# COMMAND ----------

# MAGIC %r
# MAGIC library(RGISTools)
