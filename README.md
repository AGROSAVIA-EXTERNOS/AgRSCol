 ![DataLOGO](App/logdat.JPG)
 
# Piloto: Exploración de datos para clasificación no supervisada de suelos agrícolas en regiones piloto de Colombia a partir de imágenes multiespectrales satelitales.

La caracterización y evaluación de suelos agrícolas es una labor costosa en recurso humano, capacidad técnica y tiempo, lo que evita la implementación práctica de la agricultura de precisión. Las imágenes satelitales tienen la capacidad de cubrir grandes áreas, evitando desplazamiento de personal para la recolección de información y con la capacidad de permitir explorar zonas de díficil acceso.  Sin embargo, la capacidad para distinguir características de suelo agrícolas en Colombia aún no ha sido explorada de manera exhaustiva. Desde AGROSAVIA quisiéramos evaluar de manera extensiva el potencial de imágenes satelitales multiespectrales para clasificar suelos agrícolas. Para ello usaremos máquinas de aprendizaje para realizar agrupamiento no supervisado de espectros de conjunto específicos de pixeles en áreas piloto a seleccionar por la corporación y evaluaremos los agrupamientos obtenidos. Estas herramientas podrán potencialmente ser usados como soporte para censos de uso de suelo agrícola, caracterización de propiedades físicas o químicas, e identificación de zonas con potencial de producción.


# Objetivo del proyecto

Realizar una exploración de datos inicial donde se caracterice mediante técnicas de aprendizaje no supervisado los pixeles de imágenes multiespectrales satelitales de suelos agrícolas colombianos.


# Contenido del repositorio

Para esta primera fase, incluimos los siguientes resultados preliminares

1. App                   carpeta para el Código del proyecto. En <a href=App/Fuentes-Datos-Entendimiento/pc_basics.py>pc_basics.py</a> encontraremos como realizar carga y consultas básicas usando la plataforma planetary computer de Microsoft. <a href=App/Fuentes-Datos-Entendimiento/consulta_lansat.py>consulta_lansat.py</a> Muestra una consulta de una <a href=Datos-Ejemplo/Para-Modelo/tibaitata.geojson>región de ejemplo </a> (centro investigación Tibaitatá de Agrosavia) en el catálogo de Landsat disponible a través de Planetary Computer. <a href=App/Fuentes-Datos-Entendimiento/consulta_sentinel.py>consulta_sentinel.py</a> permite hacerla para el catálogo de sentinel. La carga de una imágen multiespectral y el modelado usando clasificación no supervisada (implementación del algoritmo realizada durante el proyecto) se encuentran en <a href=App/Modelado/lansat_clustering.py> lansat_clustering.py</a>. Todos estos códigos pueden ser agregados a Databricks mediante un comando fork del repositorio.
3. Datos de Ejemplo      En <a href=Datos-Ejemplo/Para-Modelo> Datos-Ejemplo/Para-Modelo </a> encontraremos un GeoJSon de la zona de interés a estudiar y una imágen satelital de ejemplo.
4. Documentación         La documentación de los diferentes códigos se encuentran en cuadernos interactivos de python (.ipynb) en <a href=Documentacion/Model> Documentación/Model </a>





