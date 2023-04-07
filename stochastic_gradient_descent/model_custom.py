# stochastic gradient descent & badge & ridge
import numpy as np
import data

def predict_y(X, w):
    return np.dot(X, w)


def get_grad(X, Y, w, alpha, badge):
    n = Y.shape[0]
    indexs = np.random.choice(n, size = badge, replace=False)
    X_badge = X[indexs]
    Y_badge = Y[indexs]
    predicted_Y = predict_y(X_badge, w)
    return -2 * np.dot(X_badge.T, (Y_badge - predicted_Y)) + 2 * alpha * w


def predict(X, Y, w, eta, alpha, eps, badge, max_epochs):
    n = Y.shape[0]
    
    for t in range(max_epochs):
        grad = get_grad(X, Y, w, alpha, badge)
        if -eps < np.linalg.norm(grad, ord=1) < eps:
            print(t)
            return w
        else:
            eta_t = eta #/ np.sqrt(t+1)
            w = w - (eta_t * grad)
    return w


if __name__ == "__main__":
    dim = 3
    alpha = 1e-6
    badge = 3
    
    # X = data.get_X(dim = dim)
    # Y = data.get_Y()
    X = data.get_X(data.X, dim=dim)
    Y = data.Y
    w = np.zeros(dim+1)
    w = predict(X, Y, w, 0.03, alpha, 1e-4, badge, 50000)
    print(w)