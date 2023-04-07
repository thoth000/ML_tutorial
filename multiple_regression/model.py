import numpy as np
import data

def get_mean(x):
    n = len(x)
    total = 0
    for i in range(n):
        total += x[i]
    
    return total / n


def get_cov(x, y):
    n = len(x)
    mean_x = get_mean(x)
    mean_y = get_mean(y)
    total = 0
    for i in range(n):
        total += (x[i] - mean_x) * (y[i] - mean_y)
    
    return total / n


def get_var(x):
    return get_cov(x, x)


def predict(X, y):
    # 疑似逆行列を用いたパラメータ推定
    w = np.linalg.pinv(np.dot(X.T, X))
    w = np.dot(w, X.T)
    w = np.dot(w, y)
    return w


def get_r_square(X, y, w):
    predicted_y = np.dot(X, w)
    error = y - predicted_y
    return 1 - get_var(error) / get_var(y)


if __name__ == "__main__":
    dim = 3
    X = data.get_X(dim)
    Y = data.get_Y()
    
    w = predict(X, Y)
    print("w  : ", w)
    print("R^2: ", get_r_square(X, Y, w))