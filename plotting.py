import matplotlib.pyplot as plt
import os


def plot_mse_history(mse_history, filename = "mse_history.png"):
    plt.plot(mse_history)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("MSE vs Epoch")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, filename))
    plt.show()


def plot_regression_line(x, y, b0, b1, filename = "regression_fit.png"):
    plt.scatter(x, y, label = "Data")
    y_line = b0 + b1 * x
    plt.plot(x, y_line, color = "red", label = "Learned regression line")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Regression Fit")
    plt.legend()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, filename))
    plt.show()


def plot_cost_history(cost_history, filename = "cost_history.png"):
    plt.plot(cost_history)
    plt.xlabel("Epoch")
    plt.ylabel("Cost (Cross-Entropy)")
    plt.title("Cost vs Epoch")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, filename))
    plt.show()


def plot_decision_boundary(x, y, b0, b1, filename = "decision_boundary.png"):
    plt.scatter(x[y == 0], [0] * sum(y == 0), color = "blue", label = "Class 0")
    plt.scatter(x[y == 1], [1] * sum(y == 1), color = "orange", label = "Class 1")

    boundary_x = -b0 / b1
    plt.axvline(boundary_x, color = "red", linestyle = "--", label = "Decision boundary")

    plt.xlabel("x")
    plt.ylabel("Class")
    plt.title("Logistic Regression Decision Boundary")
    plt.legend()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, filename))
    plt.show()