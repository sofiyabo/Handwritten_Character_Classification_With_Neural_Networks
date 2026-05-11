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

Ximg, Yimg = load_img("./data/raw/X_images.npy","./data/raw/Y_images.npy")

img1 = Ximg[0].reshape(28, 28)
img2 = Ximg[1].reshape(28, 28)
img3 = Ximg[2].reshape(28, 28)

plt.imshow(img1, cmap='gray')
plt.axis('off')  
plt.show()

plt.imshow(img2, cmap='gray')
plt.axis('off')  
plt.show()

plt.imshow(img3, cmap='gray')
plt.axis('off')  
plt.show()

def normalization(Xtrain, Xval, Xtest):
    return Xtrain/255, Xval/255, Xtest/255