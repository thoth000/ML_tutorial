import numpy as np
import data

def predict_y(X, w):
    return np.dot(X, w)


def get_grad(X, Y, w):
    n = Y.shape[0]
    index = np.random.randint(0, n)
    predicted_y = predict_y(X[index], w)
    
    return -2 * (Y[index] - predicted_y) * X[index]


def predict(X, Y, w, eta, eps, max_epochs):
    n = Y.shape[0]
    
    for t in range(max_epochs):
        grad = get_grad(X, Y, w)
        if -eps < np.linalg.norm(grad, ord=1) < eps:
            print(t)
            return w
        else:
            eta_t = eta #/ np.sqrt(t+1)
            w = w - (eta_t * grad)
    return w


if __name__ == "__main__":
    dim = 3
    
    # X = data.get_X(dim = dim)
    # Y = data.get_Y()
    X = data.get_X(data.X, dim=dim)
    Y = data.Y
    w = np.zeros(dim+1)
    w = predict(X, Y, w, 0.03, 1e-4, 50000)
    print(w)