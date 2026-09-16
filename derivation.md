# Derivation — Gradient Descent for Simple Linear Regression

## Model

$$\hat{y}_i = b_0 + b_1 x_i$$

## Cost Function

Mean squared error over $n$ data points:

$$J(b_0, b_1) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 = \frac{1}{n}\sum_{i=1}^{n}\left(y_i - (b_0 + b_1 x_i)\right)^2$$

## Deriving ∂J/∂b₀

Let $u_i = y_i - (b_0 + b_1 x_i)$, so each term in the sum is $u_i^2$.

Apply the chain rule to differentiate the outer function:

$$\frac{\partial}{\partial b_0}(u_i^2) = 2u_i \cdot \frac{\partial u_i}{\partial b_0}$$

Differentiate the inner function $u_i = y_i - b_0 - b_1 x_i$ with respect to $b_0$. Since $y_i$ and $b_1 x_i$ don't depend on $b_0$:

$$\frac{\partial u_i}{\partial b_0} = -1$$

Combine:

$$\frac{\partial}{\partial b_0}(u_i^2) = 2u_i \cdot (-1) = -2\left(y_i - (b_0 + b_1 x_i)\right)$$

Sum over all $n$ points and divide by $n$:

$$\frac{\partial J}{\partial b_0} = -\frac{2}{n}\sum_{i=1}^{n}\left(y_i - (b_0 + b_1 x_i)\right)$$

## Deriving ∂J/∂b₁

Same setup, but differentiate the inner function with respect to $b_1$ instead:

$$\frac{\partial u_i}{\partial b_1} = -x_i$$

Combine:

$$\frac{\partial}{\partial b_1}(u_i^2) = 2u_i \cdot (-x_i) = -2x_i\left(y_i - (b_0 + b_1 x_i)\right)$$

Sum and divide by $n$:

$$\frac{\partial J}{\partial b_1} = -\frac{2}{n}\sum_{i=1}^{n} x_i\left(y_i - (b_0 + b_1 x_i)\right)$$

## Gradient Descent Update Rule

Using a learning rate $\alpha$, at each epoch:

$$b_0 \leftarrow b_0 - \alpha \frac{\partial J}{\partial b_0}$$

$$b_1 \leftarrow b_1 - \alpha \frac{\partial J}{\partial b_1}$$

## Implementation Notes

- Initial values: $b_0 = 0$, $b_1 = 0$
- Learning rate: $\alpha = 0.01$
- Epochs: 1000
- The error term $(y_i - \hat{y}_i)$ must be computed as *actual minus predicted*, matching the sign convention used throughout this derivation — using the opposite convention without also flipping the sign in the update rule would send gradient descent in the wrong direction.