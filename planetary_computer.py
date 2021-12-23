# Databricks notebook source
# MAGIC %%bash
# MAGIC pip install pystac-client

# COMMAND ----------

from pystac_client import Client
catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
collections = catalog.get_children()
for collection in collections:
    print(f"{collection.id} - {collection.title}")

# COMMAND ----------

landsat = catalog.get_child('sentinel-2-l2a')
for band in landsat.extra_fields['summaries']['eo:bands']:
    name = band['name']
    description = band['description']
    common_name = "" if 'common_name' not in band else f"({band['common_name']})"
    #ground_sample_distance = band['gsd']
    print(f"{name} {common_name}: {description} ")

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

time_range = "2000-12-01/2020-12-31"

# COMMAND ----------

search = catalog.search(
    collections=['landsat-8-c2-l2'],
    intersects=area_of_interest,
    datetime=time_range
)
print(f"{search.matched()} Items found")

# COMMAND ----------

items = list(search.items())
for item in items:
    print(f"{item.id}: {item.datetime}")

# COMMAND ----------

selected_item = items[1]

# COMMAND ----------

for asset_key, asset in selected_item.assets.items():
    print(f'{asset_key:<25} - {asset.title}')

# COMMAND ----------

import json

thumbnail_asset = selected_item.assets['thumbnail']
print(json.dumps(thumbnail_asset.to_dict(), indent=2))

# COMMAND ----------

# MAGIC %%bash
# MAGIC pip install planetary-computer

# COMMAND ----------

import planetary_computer as pc

signed_href = pc.sign(thumbnail_asset.href)

# COMMAND ----------

signed_href

# COMMAND ----------

from PIL import Image
from urllib.request import urlopen

Image.open(urlopen(signed_href))

# COMMAND ----------

from PIL import Image
from urllib.request import urlopen
import numpy as np
import matplotlib.pyplot as plt
im = Image.open(urlopen(signed_href))
img = np.array(im)
plt.imshow(img)

# COMMAND ----------

from PIL import Image
import numpy as np
from urllib.request import urlopen
pil_im = Image.open(urlopen(signed_href), 'r')
im_array = np.asarray(pil_im)
plt.imshow(im_array)
plt.show()


# COMMAND ----------

from matplotlib.pyplot import imshow
import numpy as np
from PIL import Image
from urllib.request import urlopen
import matplotlib.pyplot as plt

%matplotlib inline
pil_im = Image.open(urlopen(signed_href), 'r')
plt.imshow(np.asarray(pil_im))

