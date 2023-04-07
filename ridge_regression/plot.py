import numpy as np
import matplotlib.pyplot as plt
import model
import data

if __name__ == "__main__":
    dim = 9
    X = data.get_X(data.X, dim)
    Y = data.Y
    ploted_x = np.linspace(0, 1, 100)
    alpha_list = [1e-9, 1e-6, 1e-3, 1]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True)
    ax1.plot(data.X, data.Y, ".")
    ax2.plot(data.X_valid, data.Y_valid, ".")
    
    for alpha in alpha_list:
        w = model.predict(X, Y, alpha)
        predicted_y = model.predict_y(ploted_x, w)
        
        ax1.plot(ploted_x, predicted_y, "-", label = "alpha: {}".format(alpha))
        ax2.plot(ploted_x, predicted_y, "-", label = "alpha: {}".format(alpha))
        
        l2_norm = model.get_l2_norm(w)
        msr = model.get_msr(data.X_valid, data.Y_valid, w)
        print("alpha: {}\nL^2: {}\nMSR: {}".format(alpha, l2_norm, msr))
    
    ax1.set_title("train")
    ax2.set_title("valid")
    plt.legend()
    plt.savefig("ridge_regression")