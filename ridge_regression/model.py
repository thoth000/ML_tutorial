import numpy as np
import data

def predict(X, Y, alpha):
    w = np.linalg.pinv(np.dot(X.T, X)) + (alpha * np.identity(len(X[0])))
    w = np.dot(w, X.T)
    w = np.dot(w, Y)
    return w


def predict_y(x, w):
    dim = len(w) - 1
    X = data.get_X(x, dim)
    y = np.dot(X, w)
    return y


def get_l2_norm(w):
    return np.linalg.norm(w)


def get_msr(x, y, w):
    dim = len(w) - 1
    predicted_y = predict_y(x, w)
    error = y - predicted_y
    return np.linalg.norm(error)**2