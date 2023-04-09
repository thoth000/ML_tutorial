import numpy as np
from sklearn.preprocessing import OneHotEncoder
import plot

def images_to_vectors(X):
    X = np.reshape(X, (len(X), -1))
    return np.concatenate([X, np.ones((len(X), 1))], axis=1)


def softmax(a):
    ea = np.exp(a - np.max(a, axis=1))
    return ea / ea.sum(axis = 1)


def get_grad(x, Y, W):
    P = softmax(np.dot(W, x))
    grad = np.array(
        [
            (Y[i] - P[i]) * x for i in range(Y.shape[0])
        ]
    )
    return grad


def predict(X, Y, W, eta, max_epochs):
    for t in range(max_epochs):
        index = np.random.randint(0, Y.shape[0])
        grad = get_grad(X[index], Y[index], W)
        W = grad - eta * grad
    return W


def get_score(P, Y):
    print(P[0])
    P_indexs = P.argmax(axis = 1)
    Y_indexs = Y.argmax(axis = 1)
    
    n = P.shape[0]
    count = 0
    for i in range(n):
        # print(P_indexs[i], Y_indexs[i])
        if P_indexs[i] == Y_indexs[i]:
            count += 1

    return count / n


if __name__ == "__main__":
    encoder = OneHotEncoder(categories="auto", sparse=False)
    
    data = np.load("mnist.npz")
    X_train = images_to_vectors(data["train_x"])
    Y_train = encoder.fit_transform(data["train_y"].reshape((-1, 1)))
    params = encoder.categories_
    print(params)
    
    dim = X_train.shape[1]
    k = Y_train.shape[1]
    W = np.zeros((k, dim))
    W = predict(X_train, Y_train, W, 0.1, 50000)
    
    X_test = images_to_vectors(data["test_x"])
    Y_test = encoder.transform(data["test_y"].reshape((-1, 1)))
    P = softmax(np.dot(X_test, W.T))
    print(get_score(P, Y_test))
    