import numpy as np

def compute_predictions(x, b0, b1):
    return b0 + b1 * x

def compute_mse(y, y_pred):
    errors = y_pred - y
    return np.mean(errors**2)

def compute_gradients(x, y, b0, b1):
    n = len(x)

    y_pred = compute_predictions(x, b0, b1)
    error = y - y_pred

    db0 = -(2/n) * np.sum(error)
    db1 = -(2/n) * np.sum(x * error)

    return db0, db1

def gradient_descent(x, y, learning_rate, epochs):
    b0 = 0
    b1 = 0
    mse_history = []

    for i in range(epochs):
        y_pred = compute_predictions(x, b0, b1)
        J = compute_mse(y, y_pred)
        mse_history.append(J)

        db0, db1 = compute_gradients(x, y, b0, b1)

        b0 = b0 - learning_rate * db0
        b1 = b1 - learning_rate * db1

    return b0, b1, mse_history