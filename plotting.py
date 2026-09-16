import matplotlib.pyplot as plt
import os


def plot_mse_history(mse_history, filename="mse_history.png"):
    plt.plot(mse_history)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("MSE vs Epoch")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, filename))
    plt.show()


def plot_regression_line(x, y, b0, b1, filename="regression_fit.png"):
    plt.scatter(x, y, label="Data")
    y_line = b0 + b1 * x
    plt.plot(x, y_line, color="red", label="Learned regression line")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Regression Fit")
    plt.legend()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, filename))
    plt.show()