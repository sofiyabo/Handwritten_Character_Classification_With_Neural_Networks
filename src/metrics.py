import numpy as np

def accuracy(ypred, ytrue):
    return np.mean(ytrue == ypred)


def cross_entropy(ypred, ytrue):


def confusion_matrix(ypred, ytrue):
    n_classes = len(np.unique(ytrue))
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(ytrue, ypred):
        cm[true][pred] += 1
    
    return cm

def f1_macro(ypred, ytrue):
