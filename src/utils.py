import random
import numpy as np

def add_gaussian_noise(X, noise_level):
    """Agrega ruido gaussiano"""
    noisy = X + np.random.randn(*X.shape) * noise_level
    return np.clip(noisy, 0, 1)