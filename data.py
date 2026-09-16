import numpy as np

def generate_data(n = 100, true_b0 = 4, true_b1 = 3, noise_std = 1.0, seed = 67):
    np.random.seed(seed)
    x = np.random.uniform(0, 10, n)
    noise = np.random.normal(0, noise_std, n)
    y = true_b0 + true_b1 * x + noise
    return x, y