!pip install pyclesperanto-prototype
!pip install numpy

import numpy as np
import matplotlib as plt
import pyclesperanto_prototype as cle

from skimage.io import imread, imsave
from skimage import img_as_ubyte, io

print("Version de numpy:", np.__version__)