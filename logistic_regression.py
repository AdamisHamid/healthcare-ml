import numpy as np

def compute_predictions(x, b0, b1):
    z = b0 + b1 * x
    return 1 / (1 + np.exp(-z))

def compute_cost(y, y_pred):
    return -1 * np.mean(y * np.log(y_pred) +  (1 - y) * np.log(1 - y_pred))

def compute_gradients(x, y, b0, b1):
    n = len(x)

    y_pred = compute_predictions(x, b0, b1)
    error = y_pred - y

    db0 = np.sum(error) / n
    db1 = np.sum(x * error) / n

    return db0, db1

def gradient_descent(x, y, learning_rate, epochs):
    b0 = 0
    b1 = 0
    cost_history = []

    for i in range(epochs):
        y_pred = compute_predictions(x, b0, b1)
        J = compute_cost(y, y_pred)
        cost_history.append(J)

        db0, db1 = compute_gradients(x, y, b0, b1)

        b0 = b0 - learning_rate * db0
        b1 = b1 - learning_rate * db1

    return b0, b1, cost_history