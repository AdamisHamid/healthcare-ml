from data import generate_data
from linear_regression import gradient_descent
from plotting import plot_mse_history, plot_regression_line

x, y = generate_data()

b0, b1, mse_history = gradient_descent(x, y, learning_rate = 0.01, epochs = 1000)

print(f"Learned b0: {b0}")
print(f"Learned b1: {b1}")

plot_mse_history(mse_history)
plot_regression_line(x, y, b0, b1)