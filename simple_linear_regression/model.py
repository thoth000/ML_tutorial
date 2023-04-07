# simple regression
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


def predict(x, y):
    a = get_cov(data.X, data.Y) / get_var(data.X)
    mean_x = get_mean(data.X)
    mean_y = get_mean(data.Y)
    b = mean_y - a * mean_x
    return a, b


def get_r_square(x, y, a, b):
    predicted_y = a*x + b
    error = predicted_y - y
    # get_var(y) = get_var(predicted_y) + get_var(error)
    # return get_var(predicted_y) / get_var(y)
    return 1 - (get_var(error) / get_var(y))


if __name__ == "__main__":
    a, b = predict(data.X, data.Y)
    print("a  : ", a)
    print("b  :", b)
    print("R^2: ", get_r_square(data.X, data.Y, a, b))