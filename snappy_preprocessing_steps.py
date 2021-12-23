# Databricks notebook source
# this is a test from cesar
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import traceback
import numpy as np
import matplotlib
import shapefile
import pygeoif
import snappy
import pandas as pd
import os

from snappy import Product
from snappy import ProductIO
from snappy import ProductUtils
from snappy import WKTReader
from snappy import HashMap
from snappy import GPF
from snappy import jpy
from snappy import GeoPos
from snappy import PixelPos

# COMMAND ----------

import skimage.exposure as exposure # Correr dos veces si llega a fallar

# COMMAND ----------

pd.options.display.max_rows = 999
pd.options.display.max_colwidth = 999
pd.options.display.max_columns = 999

%matplotlib inline

# COMMAND ----------

def plot_band(band_data, gray=True):
    val1, val2 = np.percentile(band_data, (2.5,97.5))
    band_data_new = exposure.rescale_intensity(band_data, in_range=(val1,val2))

    plt.figure(figsize=(8, 8))                     
    fig = (
        plt.imshow(band_data_new, cmap = cm.gray)
        if gray else plt.imshow(band_data_new)
    )
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.show()


def get_band_array(product, band):
    band_data = product.getBand(band)
    width, height = band_data.getRasterWidth(), band_data.getRasterHeight()
    band_data_array = np.zeros(width*height, dtype = np.float32)
    band_data.readPixels(0, 0, width, height, band_data_array)
    band_data_array.shape = height, width

    print(width, height)
    return band_data_array


def categorize_bands(x):
    categories = [
        "NO_DATA", "SATURATED_DEFECTIVE", "DARK_FEATURE_SHADOW",
        "CLOUD_SHADOW", "VEGETATION", "NOT_VEGETATION",
        "WATER", "UNCLASSIFIED", "CLOUD_MEDIUM_PROBA",
        "CLOUD_HIGH_PROBA", "THIN_CIRRUS", "SNOW_ICE"
    ]
    
    for i in range(len(categories)):
        if x == i:
            return categories[i]

        
def get_pixel_into_df(df, product):
    
    gc = product.getSceneGeoCoding()
    bands_names = list(product.getBandNames())
    
    cols = ['X', 'Y']
    cols.extend(df.columns)
    cols.extend(bands_names)
    
    df_bands = pd.DataFrame(columns=cols)
    rows = df.shape[0]
    
    for j in range(rows):
        
        lat, lon = df.loc[j][['lat', 'lon']]
        pixel_pos = gc.getPixelPos(GeoPos(lat, lon), None)

    #     print("(lat, lon) -→ (X, Y) : (%s, %s) -→ (%s, %s)" % (lat, lon, int(pixel_pos.x), int(pixel_pos.y)))

        data = [int(pixel_pos.x), int(pixel_pos.y)]
        
        for col in df.columns:
            data.append(df.loc[j, col])

        for i, band_name in enumerate(bands_names):

            temp_band = product.getBand(band_name)
            width, height = temp_band.getRasterWidth(), temp_band.getRasterHeight()
            try:
                tmp = np.zeros(1)
                temp_band.readPixels(int(pixel_pos.x), int(pixel_pos.y), 1, 1, tmp)
                data.append(tmp[0])
            except Exception as e:
                print(band_name)
                print(width, height)
                print(int(pixel_pos.x), int(pixel_pos.y))
                data.append(-9999999)
                print(e)
                traceback.print_exc()
#         print(len(data))
        df_bands.loc[df_bands.shape[0]] = data
    return df_bands


def get_operator_info(name):

    op_spi = GPF.getDefaultInstance().getOperatorSpiRegistry().getOperatorSpi(name)

    print('Operator name: %s' % op_spi.getOperatorDescriptor().getName())
    print('Operator alias: %s\n' % op_spi.getOperatorDescriptor().getAlias())

    param_Desc = op_spi.getOperatorDescriptor().getParameterDescriptors()
    for param in param_Desc:
        info = (
            "Name \"%s\": {\n\tAlias: %s,\n\tdtype: %s,\n\tdefault value: %s,\n\tdescription: %s\n}\n" %
            (
                param.getName(), param.getAlias(), param.getDataType().getName(),
                param.getDefaultValue(), param.getDescription()
            )
        )
        print(info)
    #     print(help(param.getDataType())) 

# COMMAND ----------

HashMap = snappy.jpy.get_type('java.util.HashMap')
GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()

# COMMAND ----------

# MAGIC %md
# MAGIC ---

# COMMAND ----------

ls "../../data/raw/sentinel_satellite_images/"

# COMMAND ----------

ls "../../data/processed_raw/sentinel_satellite_images/"

# COMMAND ----------

# MAGIC %md
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read product

# COMMAND ----------

path_to_sentinel_data = "../S2A_MSIL1C_20151228T151652_N0201_R125_T18NXM_20151228T151649.zip"
product = ProductIO.readProduct(path_to_sentinel_data)

# COMMAND ----------

print(list(product.getBandNames()))

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ### Atmospheric correction

# COMMAND ----------

get_operator_info('Sen2Cor255')

# COMMAND ----------

# cmd = ‘Sen2Cor-02.05.05-win64\L2A_Process ‘+path+’ --resolution=60’
# subprocess.call(cmd, shell=True)

# COMMAND ----------

parameters = HashMap()

parameters.put("resolution", "ALL") # 10, 20, 60
# parameters.put("targetProductFile", "../../data/processed_raw/")
product_atm_corr = GPF.createProduct("Sen2Cor255", parameters, product) # Sen2Cor255

# COMMAND ----------

print(list(product_atm_corr.getBandNames()))

# COMMAND ----------

# filename_out = 'S2A_MSIL2A_20160110T152632_N0201_R025_T18NXM_20160110T153105.dim'
# ProductIO.writeProduct(product_ac2, filename_out, 'BEAM-DIMAP')

# COMMAND ----------

b4_array = get_band_array(product_atm_corr, 'B4')

# COMMAND ----------

plot_band(b4_array)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ### Resampling

# COMMAND ----------

get_operator_info('Resample')

# COMMAND ----------

parameters_resample = HashMap()
target_resolution = 10 # 20, 60
upsampling_method = 'Nearest'
downsampling_method = 'Mean'
parameters_resample.put('downsamplingMethod', downsampling_method)
parameters_resample.put('upsamplingMethod', upsampling_method)
parameters_resample.put('targetResolution', target_resolution)

product_resample = snappy.GPF.createProduct('Resample', parameters_resample, product_atm_corr)

# COMMAND ----------

B4 = product_resample.getBand('B4')
B5 = product_resample.getBand('B5')

width_B4, height_B4 = B4.getRasterWidth(), B4.getRasterHeight()
width_B5, height_B5 = B5.getRasterWidth(), B5.getRasterHeight()

print("Band 4 Size: %s, %s" % (width_B4, height_B4))
print("Band 5 Size: %s, %s" % (width_B5, height_B5))

# COMMAND ----------

b8_array = get_band_array(product_resample, 'B8')

# COMMAND ----------

plot_band(b8_array)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ### Subsetting

# COMMAND ----------

get_operator_info('Subset')

# COMMAND ----------

geometry = WKTReader().read("POLYGON ((-73.7 5.85, -73.3 5.85, -73.3 6.2, -73.7 6.2, -73.7 5.85))")

# COMMAND ----------

parameters = HashMap()
parameters.put('copyMetadata', True)
parameters.put('geoRegion', geometry)
product_subset = snappy.GPF.createProduct('Subset', parameters, product_resample)

# COMMAND ----------

b6_array = get_band_array(product_subset, 'B6')

# COMMAND ----------

plot_band(b6_array)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ### Cloud masking, Indices and ratios

# COMMAND ----------

get_operator_info('BandMaths')

# COMMAND ----------

names = [
    'CloudMask',
    'NDVI',
    'GNDVI',
    'EVI',
    'AVI',
    'SAVI',
    'NDMI',
    'MSI',
    'GCI',
    'NBR',
    'BSI',
    'NDWI',
    'NDSI',
    'NDGI',
    'ARVI',
    'SIPI'
]
expresions = [
    'if (scl_cloud_medium_proba || scl_cloud_high_proba || scl_thin_cirrus) then 1 else 0',
    '(B8-B4) / (B8+B4)',
    '(B8-B3) / (B8+B3)',
    '2.5*( (B8-B4) / ( (B8 + (6*B4) - (7.5*B2) + 1) ) )',
    'pow( B8 * (1-B4) * (B8-B4), 1/3)',
    '1.428 * ( (B8-B4) / (B8+B4+0.428) )',
    '(B8-B11) / (B8+B11)',
    'B11 / B8',
    '(B9 / B3) - 1',
    '(B8-B12) / (B8+B12)',
    '((B11+B4) - (B8+B2)) / ((B11+B4) + (B8+B2))',
    '(B3-B8) / (B3+B8)',
    '(B3-B11) / (B3+B11)',
    '(B8-B3) / (B8+B3)',
    '(B8 - (2*B4) + B2) / (B8 + (2*B4) + B2)',
    '(B8-B2) / (B8+B2)',
]

inds_ = [i for i in range(len(names))]

print(len(names) == len(expresions))

# COMMAND ----------

paramsCloud = HashMap()

targetBands = jpy.array(
    'org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor',
    len(names)
)
BandDescriptor = jpy.get_type(
    'org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor'
)

for i, name, expresion in zip(inds_, names, expresions):
    targetBand = BandDescriptor()
    targetBand.name = name
    targetBand.type = 'float32'
    targetBand.expression = expresion
    targetBands[i] = targetBand

paramsCloud.put('targetBands', targetBands)
productCloud = GPF.createProduct('BandMaths', paramsCloud, product_subset)

# COMMAND ----------

print(list(productCloud.getBandNames()))

# COMMAND ----------

NDVI_array = get_band_array(productCloud, 'NDVI')
plot_band(NDVI_array, False)

# COMMAND ----------

NDWI_array = get_band_array(productCloud, 'NDWI')
plot_band(NDWI_array, False)

# COMMAND ----------

BSI_array = get_band_array(productCloud, 'BSI')
plot_band(BSI_array, False)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ### Merge products

# COMMAND ----------

get_operator_info('Merge')

# COMMAND ----------

sourceProducts= HashMap()
sourceProducts.put('masterProduct', product_subset)
sourceProducts.put('slaveProduct', productCloud)
parametersMerge = HashMap()
productMerged = GPF.createProduct('Merge', parametersMerge, sourceProducts)

# COMMAND ----------

# parameters = HashMap()
# target = GPF.createProduct("BandMerge", parameters, (product_subset, product_subset_ndvi))

# COMMAND ----------

print(list(productMerged.getBandNames()))

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ### Save product

# COMMAND ----------

filename_out = '../../data/processed_raw/sentinel_satellite_images/S2A_MSIL2A_20160110T152632_N0201_R025_T18NXM_20160110T153105_PROCESSED.dim'
ProductIO.writeProduct(productMerged, filename_out, 'BEAM-DIMAP')

# COMMAND ----------


# def createProgressMonitor():
#     PWPM = jpy.get_type('com.bc.ceres.core.PrintWriterProgressMonitor')
#     JavaSystem = jpy.get_type('java.lang.System')
#     monitor = PWPM(JavaSystem.out)
#     return monitor

# incremental = False
# pm = createProgressMonitor()
# filename_out = 'S2A_MSIL2A_20160110T152632_N0201_R025_T18NXM_20160110T153105_subsetoutput3.dim'
# GPF.writeProduct(product_subset , 'S2A_MSIL2A_20160110T152632_N0201_R025_T18NXM_20160110T153105_subsetoutput2.dim', 'BEAM-DIMAP', incremental, pm)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## End of processing
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ### Extract pixel values

# COMMAND ----------

path_to_sentinel_data = (
    "../../data/processed_raw/sentinel_satellite_images/" +
    "S2A_MSIL2A_20160110T152632_N0201_R025_T18NXM_20160110T153105_PROCESSED.dim"
)
product_pixel_extraction = ProductIO.readProduct(path_to_sentinel_data)

# COMMAND ----------

print(list(product_pixel_extraction.getBandNames()))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC #### Soil_data

# COMMAND ----------

path_nirs = os.path.abspath(
    "../../data/processed_raw/soil_cane_vis_NIRS/soil_cane_vis_NIR_georeferenced.csv"
)
soil_nir_data = pd.read_csv(path_nirs, sep=';')
print(soil_nir_data.shape)
soil_nir_data.head()

# COMMAND ----------



# COMMAND ----------

df_soil_nir_data_pixels = get_pixel_into_df(soil_nir_data, product_pixel_extraction)
print(df_soil_nir_data_pixels.shape)
df_soil_nir_data_pixels.head()

# COMMAND ----------

df_soil_nir_data_pixels['quality_scene_classification'].unique()

# COMMAND ----------

df_soil_nir_data_pixels['quality_classification'] = (
    df_soil_nir_data_pixels['quality_scene_classification'].apply(lambda x: categorize_bands(x))
)

# COMMAND ----------

unique_qualities = df_soil_nir_data_pixels['quality_scene_classification'].unique()
unique_qualities

# COMMAND ----------

for i in unique_qualities:
    df_temp = df_soil_nir_data_pixels[
        df_soil_nir_data_pixels['quality_scene_classification'] == i
    ]
    print("%s count: %s" %(df_temp.iloc[0]['quality_classification'], df_temp.shape[0]))

# COMMAND ----------

df_soil_nir_data_pixels.to_csv(
    '../../data/processed_raw/sentinel_bands/' +
    'S2A_MSIL2A_20160110T152632_N0201_R025_T18NXM_20160110T153105_PROCESSED_SOIL.csv',
    sep=';', index=False
)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC #### Vegetal data

# COMMAND ----------

path_nirs = os.path.abspath(
    "../../data/processed_raw/soil_cane_vis_NIRS/remuestreo_2020_muestra_tejido.csv"
)
vegetal_data = pd.read_csv(path_nirs, sep=';')
print(vegetal_data.shape)
vegetal_data.head()

# COMMAND ----------

df_vegetal_data_pixels = get_pixel_into_df(vegetal_data, product_pixel_extraction)
print(df_vegetal_data_pixels.shape)
df_vegetal_data_pixels.head()

# COMMAND ----------

unique_qualities = df_vegetal_data_pixels['quality_scene_classification'].unique()
unique_qualities

# COMMAND ----------

df_vegetal_data_pixels['quality_classification'] = (
    df_vegetal_data_pixels['quality_scene_classification'].apply(lambda x: categorize_bands(x))
)

# COMMAND ----------

for i in unique_qualities:
    df_temp = df_vegetal_data_pixels[
        df_vegetal_data_pixels['quality_scene_classification'] == i
    ]
    print("%s count: %s" %(df_temp.iloc[0]['quality_classification'], df_temp.shape[0]))

# COMMAND ----------

df_vegetal_data_pixels.to_csv(
    '../../data/processed_raw/sentinel_bands/' +
    'S2A_MSIL2A_20160110T152632_N0201_R025_T18NXM_20160110T153105_PROCESSED_VEGETATION.csv',
    sep=';', index=False
)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ### PixEx operator for pixel extraction

# COMMAND ----------

get_operator_info('PixEx')

# COMMAND ----------

my_coordinates = jpy.array('org.esa.snap.pixex.Coordinate', soil_nir_data.shape[0])

for j in range(soil_nir_data.shape[0]):
#     lat, lon = (5.914702982, -73.51362747)
    lat, lon = soil_nir_data.loc[j][['lat', 'lon']]
#     lat, lon = (5.914703, -73.513627)

    coor = jpy.get_type('org.esa.snap.pixex.Coordinate')
    my_coordinates[j] = coor('bin%s' % j, lat, lon, None)
#     print(my_coordinates)

parameters = HashMap()
parameters.put('exportBands', True)
parameters.put('exportExpressionResult', False)
parameters.put('exportMasks', False)
parameters.put('exportTiePoints', False)
parameters.put('outputDir', '../../data/processed_raw/sentinel_bands/PixEx')
parameters.put('coordinates', my_coordinates)

GPF.createProduct('PixEx', parameters, product_pixel_extraction)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ### Compare PixEx and readPixels for pixel extraction

# COMMAND ----------

df_pixex = pd.read_csv(
    '../../data/processed_raw/sentinel_bands/PixEx/pixEx_S2_MSI_Level-2Ap_measurements.txt',
    sep='\t', skiprows=6
)
print(df_pixex.shape) ## it didn't import all coordinates
df_pixex.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ---

# COMMAND ----------

df_read_pixel = pd.read_csv(
    '../../data/processed_raw/sentinel_bands/' +
    'S2A_MSIL2A_20160110T152632_N0201_R025_T18NXM_20160110T153105_processed.csv',
    sep=';'
)
print(df_read_pixel.shape)
df_read_pixel.head()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


