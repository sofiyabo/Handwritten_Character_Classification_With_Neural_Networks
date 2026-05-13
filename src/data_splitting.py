import numpy as np

def initial_split(X, y):
    s = np.random.seed(46)

    #el split va a ser 70/15/15

    n = len(X)
    ind = np.random.permutation(n) # shuffle de los indices

    train_end = int(0.7 * n)
    val_end = int(0.85 *n)

    train_ind = ind[:train_end]
    Xtrain = X[train_ind]
    Ytrain = y[train_ind]

    val_ind = ind[train_end:val_end]
    Xval = X[val_ind]
    Yval = y[val_ind]

    test_ind = ind[val_end:]
    Xtest = X[test_ind]
    Ytest = y[test_ind]

    return Xtrain, Ytrain, Xval, Yval, Xtest, Ytest


