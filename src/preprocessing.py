import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_img(pathX, pathY):
    X_images = np.load(pathX)
    Y_images = np.load(pathY)

    return X_images, Y_images

def normalization(Xtrain, Xval, Xtest):
    return Xtrain/255, Xval/255, Xtest/255