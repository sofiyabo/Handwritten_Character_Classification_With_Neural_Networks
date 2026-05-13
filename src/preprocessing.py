import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys 
import os



sys.path.append(os.path.abspath("../src"))
sys.path.append(os.path.abspath("../data"))

#El path va a cambiar por un punto cuando cambie al notebook

def load_img(pathX, pathY):
    X_images = np.load(pathX)
    Y_images = np.load(pathY)

    return X_images, Y_images

def normalization(Xtrain, Xval, Xtest):
    return Xtrain/255, Xval/255, Xtest/255