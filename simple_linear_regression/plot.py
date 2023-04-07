import numpy as np
import matplotlib.pyplot as plt
import model
import data

if __name__ == "__main__":
    a, b = model.predict(data.X, data.Y)
    n = len(data.X)
    fig, ax = plt.subplots()
    
    ax.plot(data.X, data.Y, ".")
    x = np.array([0, 10])
    ax.plot(x, a*x+b, "-", label = "line: {:.3}x + {:.3}".format(a, b))
    ax.set_xlim(x)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid()
    ax.legend()
    plt.savefig("simple_linear_regression.png")