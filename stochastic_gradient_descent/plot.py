import numpy as np
import matplotlib.pyplot as plt
import model_basic, model_badge, model_ridge, model_custom, model_gd
import data
if __name__ == "__main__":
    dim = 3
    alpha = 1e-6
    eta = 0.05
    badge = 4
    
    x = data.X
    X = data.get_X(x, dim)
    Y = data.Y
    models = []
    names  = ["basic", "badge", "ridge", "custom", "GD"]
    
    # basic
    w = np.zeros(dim+1)
    models.append(model_basic.predict(X, Y, w, eta, 1e-4, 100000))
    # badge
    w = np.zeros(dim+1)
    models.append(model_badge.predict(X, Y, w, eta, 1e-4, badge, 100000))
    # ridge
    w = np.zeros(dim+1)
    models.append(model_ridge.predict(X, Y, w, eta, alpha, 1e-4, 100000))
    # custom(badge & ridge)
    w = np.zeros(dim+1)
    models.append(model_custom.predict(X, Y, w, eta, alpha, 1e-4, badge, 100000))
    # GD
    w = np.zeros(dim+1)
    models.append(model_gd.predict(X, Y, w, eta, 1e-4, 100000))
    
    # plot
    fig, ax = plt.subplots()
    ax.plot(data.X, Y, ".")
    ploted_x = np.linspace(0, 1., 100)
    X = data.get_X(ploted_x, dim)
    for i in range(len(models)):
        name = names[i]
        w = models[i]
        print(w)
        predicted_Y = model_basic.predict_y(X, w)
        ax.plot(ploted_x, predicted_Y, label = name)
    ax.set_title("SGD")
    ax.legend()
    plt.savefig("images/stochastic_gradient_descent_dim{}.png".format(dim))