import numpy as np

def compute_predictions(x, b0, b1):
    return b0 + b1 * x

def compute_mse(y, y_pred):
    errors = y_pred - y
    return np.mean(errors**2)