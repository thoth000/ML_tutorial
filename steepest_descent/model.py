import numpy as np
import data

def predict_y(X, w):
    return np.dot(X, w)


def get_grad(X, y, w):
    predicted_y = predict_y(X, w)
    return -2 * np.dot(X.T, (y - predicted_y))


def predict(X, y, w, eta, eps, max_epochs):
    for t in range(max_epochs):
        grad = get_grad(X, y, w)
        if -eps < np.linalg.norm(grad, ord = 1) < eps:
            print(t)
            return w
        w = w - eta*grad
    return w


if __name__ == "__main__":
    dim = 3
    # X = data.get_X(dim=dim)
    # Y = data.get_Y()
    X = data.get_X(data.X, dim=dim)
    Y = data.Y
    w = np.zeros(X.shape[1])
    w = predict(X, Y, w, 0.001, 1e-4, 10000)
    print(w)