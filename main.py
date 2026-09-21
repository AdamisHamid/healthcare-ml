from data import generate_data, generate_classification_data
from linear_regression import gradient_descent as linear_gradient_descent
from logistic_regression import gradient_descent as logistic_gradient_descent
from plotting import plot_mse_history, plot_regression_line, plot_cost_history, plot_decision_boundary

# --- Linear Regression ---
x, y = generate_data()

b0, b1, mse_history = linear_gradient_descent(x, y, learning_rate = 0.01, epochs = 1000)

print(f"Linear — Learned b0: {b0}")
print(f"Linear — Learned b1: {b1}")

plot_mse_history(mse_history)
plot_regression_line(x, y, b0, b1)

# --- Logistic Regression ---
x_class, y_class = generate_classification_data()

b0_class, b1_class, cost_history = logistic_gradient_descent(x_class, y_class, learning_rate = 0.1, epochs = 1000)

print(f"Logistic — Learned b0: {b0_class}")
print(f"Logistic — Learned b1: {b1_class}")

plot_cost_history(cost_history)
plot_decision_boundary(x_class, y_class, b0_class, b1_class)