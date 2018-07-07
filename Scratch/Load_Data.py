import os
import pickle
import numpy as np
from scipy.ndimage import rotate 

def load_CIFAR_batch(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
    return X, Y

def load_CIFAR10(ROOT):
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)    
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
    
    cifar10_dir = './cifar-10-python'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    mask = range(num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]

    mask = range(num_validation, num_training + num_validation)
    X_train = X_train
    y_train = y_train

    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    
    X_train  /= np.std(X_train,axis=0)
    X_test /= np.std(X_test,axis=0)
    X_val /= np.std(X_val,axis=0)

    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }