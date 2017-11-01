import numpy as np
import computeCost as cc

def gradient_descent(X, y, theta, alpha, num_iters) :
    m = y.shape[0]
    cost_hostory = np.zeros((num_iters,1))
    for i in range(num_iters):
        sigma = np.dot(X.T, (np.dot(X, theta) - y))
        theta = theta - alpha/m*sigma;
        cost_hostory[i] = cc.compute_cost(X, y, theta)
    return theta, cost_hostory