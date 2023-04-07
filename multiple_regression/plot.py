import numpy as np
import matplotlib.pyplot as plt
import model
import data

if __name__ == "__main__":
    dim = 4
    x_start = 0
    x_end = 10
    
    X = data.get_X(dim)
    x = X[:, 1]
    Y = data.get_Y()
    w = model.predict(X, Y)
    r_square = model.get_r_square(X, Y, w)
    
    fig, ax = plt.subplots()
    ax.plot(x, Y, ".")
    x = np.linspace(x_start, x_end, 100)
    X = data.get_X(dim, x)
    ax.plot(x, np.dot(X, w), "-", label = "R^2: {}".format(r_square))
    
    ax.set_title("multiple_regression_{}dim".format(dim))
    ax.set_xlim(x_start, x_end)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid()
    ax.legend()
    
    plt.savefig("multiple_regression_{}dim.png".format(dim))