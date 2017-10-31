import numpy as np

def compute_cost(X, y, theta) :
    m = y.shape[0]
    return np.sum(np.power((np.dot(X, theta) - y), 2)/m/2)