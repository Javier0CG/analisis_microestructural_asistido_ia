!pip install pyclesperanto-prototype
!pip install numpy

import numpy as np
import matplotlib as plt
import pyclesperanto_prototype as cle
import pandas as pd
import seaborn as sns

from skimage.io import imread, imsave
from skimage import img_as_ubyte, io

print("Version de numpy:", np.__version__)
print("Version de matplotlib:", plt.__version__)
print("Version de pyclesperanto_prototype:", cle.__version__)

#aqui debes introducir la ruta de tu imagen
img_path = "/content/micro.jpg" 

#leer la img
imagen_original = io.imread(img_path)

#img en gris
imagen_original_gris = img_as_ubyte(io.imread(img_path, as_gray=True))

#invertir img
imagen_invertida = np.invert(imagen_original_gris)

#binarizar img
imagen_binarizada_original = cle.binary_not(cle.threshold_otsu(imagen_original_gris))
imagen_binarizada_invertida = cle.binary_not(cle.threshold_otsu(imagen_invertida))

#segmentar granos
img_granos_etiquetados = cle.voronoi_labeling(imagen_binarizada_invertida)
cle.imshow(img_granos_etiquetados, labels=True)

#segmentar granos sin los que tocan el borde
img_granos_etiquetados_sinborde = cle.exclude_labels_on_edges(img_granos_etiquetados)
cle.imshow(img_granos_etiquetados_sinborde, labels=True)

#contar los granos
num_granos = cle.maximum_of_all_pixels(img_granos_etiquetados_sinborde)
print("Numero de granos: ", num_granos)

#estadistica de granos e histograma
estadistica_granos = cle.statistics_of_labelled_pixels(imagen_original, img_granos_etiquetados_sinborde)
tabla_estadistica = pd.DataFrame(estadistica_granos)
#print(tabla_estadistica.info())
sns.histplot(tabla_estadistica['area'], fill=True)

print("done")