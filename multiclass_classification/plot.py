import numpy as np
import matplotlib.pyplot as plt

def show_image(x, y):
    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(1,1,1)
    ax.set_title('Gold label: {}'.format(y))
    im = ax.imshow(x)
    fig.colorbar(im)
    plt.show()


def show_graph(x):
    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(1,1,1)
    ax.plot(range(len(x)), x)
    ax.set_xlabel('Position')
    ax.set_ylabel('Brightness')
    plt.show()


if __name__ == "__main__":
    data = np.load("mnist.npz")
    index = 0
    x = data["train_x"][index]
    y = data["train_y"][index]
    
    show_image(x, y)