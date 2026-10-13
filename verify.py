import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

from data import generate_data, generate_classification_data
from linear_regression import gradient_descent as linear_gd, compute_mse
from logistic_regression import gradient_descent as logistic_gd, compute_cost, compute_predictions as logistic_predict


def row(name, ours, ref):
    print(f"  {name:<10} ours = {ours:>9.5f}   sklearn = {ref:>9.5f}   |diff| = {abs(ours - ref):.2e}")


# ---------------------------------------------------------------- linear
print("=== Linear regression (same settings as main.py: lr 0.01, 1000 epochs)")
x, y = generate_data()
b0, b1, mse_hist = linear_gd(x, y, learning_rate = 0.01, epochs = 1000)

ols = LinearRegression().fit(x.reshape(-1, 1), y)
row("b0", b0, ols.intercept_)
row("b1", b1, ols.coef_[0])
mse_ours = compute_mse(y, b0 + b1 * x)
mse_ols = compute_mse(y, ols.predict(x.reshape(-1, 1)))
print(f"  MSE        ours = {mse_ours:.6f}   sklearn = {mse_ols:.6f}")

print("\n  Convergence check: more epochs, same learning rate")
for epochs in [1000, 5000, 20000]:
    c0, c1, _ = linear_gd(x, y, learning_rate = 0.01, epochs = epochs)
    print(f"  {epochs:>6} epochs: |b0 diff| = {abs(c0 - ols.intercept_):.2e}, |b1 diff| = {abs(c1 - ols.coef_[0]):.2e}")

# ---------------------------------------------------------------- logistic
print("\n=== Logistic regression (same settings as main.py: lr 0.1, 1000 epochs)")
xc, yc = generate_classification_data()
X = xc.reshape(-1, 1)
print(f"  Class balance: {yc.sum()} positives out of {len(yc)}")

c0, c1, cost_hist = logistic_gd(xc, yc, learning_rate = 0.1, epochs = 1000)

mle = LogisticRegression(penalty = None, tol = 1e-10, max_iter = 10000).fit(X, yc)
ridge = LogisticRegression().fit(X, yc)

r0, r1 = mle.intercept_[0], mle.coef_[0, 0]
row("b0", c0, r0)
row("b1", c1, r1)
row("boundary", -c0 / c1, -r0 / r1)

def loss(b0, b1):
    p = logistic_predict(xc, b0, b1)
    return compute_cost(yc, p)

def acc(b0, b1):
    return np.mean((logistic_predict(xc, b0, b1) >= 0.5) == yc)

print(f"  Log-loss   ours = {loss(c0, c1):.6f}   sklearn = {loss(r0, r1):.6f}")
print(f"  Accuracy   ours = {acc(c0, c1):.3f}      sklearn = {acc(r0, r1):.3f}")
print(f"  Cost still falling at epoch 1000? last 100 epochs dropped by {cost_hist[-100] - cost_hist[-1]:.2e}")

print("\n  sklearn default (L2 penalty, C=1.0), shown to explain why we don't compare against it:")
print(f"  b0 = {ridge.intercept_[0]:.5f}, b1 = {ridge.coef_[0, 0]:.5f}, boundary = {-ridge.intercept_[0] / ridge.coef_[0, 0]:.5f}")

print("\n  Convergence check: more epochs, same learning rate")
for epochs in [1000, 10000, 50000, 200000]:
    d0, d1, _ = logistic_gd(xc, yc, learning_rate = 0.1, epochs = epochs)
    print(f"  {epochs:>6} epochs: b0 = {d0:.4f}, b1 = {d1:.4f}, |b0 diff| = {abs(d0 - r0):.2e}, |b1 diff| = {abs(d1 - r1):.2e}")
