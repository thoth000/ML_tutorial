import numpy as np
from sklearn.model_selection import train_test_split  # 訓練データ, 検証データの分割
from sklearn.preprocessing import LabelEncoder        # ラベルにクラス番号を割り当て
from sklearn.feature_extraction import DictVectorizer # 辞書オブジェクトから特徴ベクトルへの変換
import data

def sigmoid(a):
    # return 1 / (1 + np.exp(-a))
    if a >= 0:
        return 1 / (1 + np.exp(-a))
    else:
        return 1 - 1 / (1 + np.exp(a))


def predict_p(x, w):
    return sigmoid(np.dot(x.T, w))


def get_grad(x, y, w):
    p = predict_p(x, w)
    return -(y - p) * x


def predict(X, Y, w, eta, eps, max_epochs):
    n = X.shape[0]
    for t in range(max_epochs):
        index = np.random.randint(0, n)
        x_badge = X[index]
        y_badge = Y[index]
        grad = get_grad(x_badge, y_badge, w)
        if np.linalg.norm(grad, ord=1) < eps:
            return w
        else:
            w = w - eta * grad
    return w


def add_zero_dimension(X):
    n = X.shape[0]
    
    zero_dimensions = np.ones((n, 1))
    result = np.concatenate((X, zero_dimensions), axis=1)
    return result


def get_score(P, Y):
    n = Y.shape[0]
    P = np.round(P)
    count = 0
    
    for i in range(n):
        if P[i] == Y[i]:
            count += 1
    return count / n


if __name__ == "__main__":
    D = data.get_data("SMSSpamCollection")
    D_train, D_test = train_test_split(D, test_size=0.1, random_state=0)
    
    VX = DictVectorizer(sparse=False)
    VY = LabelEncoder()
    
    X_train = add_zero_dimension(VX.fit_transform([d[0] for d in D_train]))
    Y_train = VY.fit_transform([d[1] for d in D_train])
    X_test = add_zero_dimension(VX.transform([d[0] for d in D_test]))
    Y_test = VY.transform([d[1] for d in D_test])
    # print(D_train[10])
    # print(X_train[10])
    # print(VX.feature_names_[12653])
    # print(VY.classes_)
    
    dim = X_train.shape[1]
    w = np.zeros(dim)
    w = predict(X_train, Y_train, w, 0.05, 1e-4, 50000)
    
    predicted_P = np.array([predict_p(x, w) for x in X_test])
    score = get_score(predicted_P, Y_test)
    print("正解率A: {}\n".format(score))
    
    F = np.array(sorted(zip(VX.feature_names_, w), key=lambda x: np.abs(x[1])))
    print(F[-20:]) # 影響の大きい20要素を出力