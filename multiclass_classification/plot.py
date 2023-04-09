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


def show_weight_image(w):
    fig, ax = plt.subplots(dpi = 100)
    ax.set_aspect("equal")
    ax.xaxis.tick_top()
    im = ax.imshow(w)
    fig.colorbar(im, ax=ax)
    plt.show()


def save_weight_images(W):
    fig, axes = plt.subplots(2, 5, tight_layout=True)
    for i in range(2):
        for j in range(5):
            number = i*5 + j
            w = W[number]
            axes[i, j].set_aspect("equal")
            axes[i, j].xaxis.tick_top()
            axes[i, j].axis("off")
            axes[i, j].set_title(number)
            im = axes[i, j].imshow(w)
    plt.show()


if __name__ == "__main__":
    data = np.load("mnist.npz")
    index = 0
    x = data["train_x"][index]
    y = data["train_y"][index]
    
    show_image(x, y)