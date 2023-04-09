import numpy as np
from sklearn.linear_model import SGDClassifier
import plot

def images_to_vectors(X):
    return np.reshape(X, (X.shape[0], -1))


if __name__ == "__main__":
    data = np.load('mnist.npz')
    X_train = images_to_vectors(data['train_x'])
    Y_train = data['train_y']
    X_test = images_to_vectors(data['test_x'])
    Y_test = data['test_y']   

    model = SGDClassifier(loss="log")
    model.fit(X_train, Y_train)
    
    # print(Y_test[0])
    # print(model.predict(X_test[0: 1])[0])
    
    score = model.score(X_test, Y_test)
    print("score: {}".format(score))
    
    # number = 0
    # w = model.coef_[number].reshape(28, 28)
    W = model.coef_
    plot.save_weight_images(W.reshape(W.shape[0], 28, 28))