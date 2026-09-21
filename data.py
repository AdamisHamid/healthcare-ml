import numpy as np

def generate_data(n = 100, true_b0 = 4, true_b1 = 3, noise_std = 1.0, seed = 67):
    np.random.seed(seed)
    x = np.random.uniform(0, 10, n)
    noise = np.random.normal(0, noise_std, n)
    y = true_b0 + true_b1 * x + noise
    return x, y

def generate_classification_data(n = 100, true_b0 = -7.5, true_b1 = 1.5, seed = 67):
    np.random.seed(seed)
    x = np.random.uniform(0, 10, n)
    z = true_b0 + true_b1 * x
    true_prob = 1 / (1 + np.exp(-z))
    y = np.random.binomial(1, true_prob)
    return x, y