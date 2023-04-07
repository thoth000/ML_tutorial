import numpy as np

D = np.array([[1, 3], [3, 6], [6, 5], [8, 7]])

def get_X(dim, x = D[:, 0]):
    n = len(x)
    X = np.ones((n, dim + 1))
    for i in range(1, dim+1):
        tmp_x = x**i
        for j in range(0, n):
            X[j, i] = tmp_x[j]

    return X

def get_Y():
    return D[:, 1]