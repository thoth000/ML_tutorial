import numpy as np
import data

def predict_y(X, w):
    return np.dot(X, w)


def get_grad(X, Y, w, alpha):
    n = Y.shape[0]
    index = np.random.randint(0, n)
    predicted_y = predict_y(X[index], w)
    return -2 * np.dot(X[index].T, Y[index] - predicted_y) + 2 * alpha * w

def predict(X, Y, w, eta, alpha, eps, max_epochs):
    n = Y.shape[0]
    
    for t in range(max_epochs):
        grad = get_grad(X, Y, w, alpha)
        if -eps < np.linalg.norm(grad, ord=1) < eps:
            print(t)
            return w
        else:
            eta_t = eta #/ np.sqrt(t+1)
            w = w - (eta_t * grad)
        # print(w)
    return w


if __name__ == "__main__":
    dim = 1
    alpha = 1e-6
    
    X = data.get_X(dim = dim)
    Y = data.get_Y()
    w = np.zeros(dim+1)
    w = predict(X, Y, w, 0.03, alpha, 1e-4, 50000)
    print(w)