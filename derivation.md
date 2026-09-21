# Derivation

## Part 1: Linear Regression

### Model

$$\hat{y}_i = b_0 + b_1 x_i$$

### Cost Function

Mean squared error over $n$ data points:

$$J(b_0, b_1) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 = \frac{1}{n}\sum_{i=1}^{n}\left(y_i - (b_0 + b_1 x_i)\right)^2$$

### Deriving $\frac{\partial J}{\partial b_0}$

Let $u_i = y_i - (b_0 + b_1 x_i)$, so each term in the sum is $u_i^2$.

Apply the chain rule to differentiate the outer function:

$$\frac{\partial}{\partial b_0}(u_i^2) = 2u_i \cdot \frac{\partial u_i}{\partial b_0}$$

Differentiate the inner function $u_i = y_i - b_0 - b_1 x_i$ with respect to $b_0$. Since $y_i$ and $b_1 x_i$ don't depend on $b_0$:

$$\frac{\partial u_i}{\partial b_0} = -1$$

Combine:

$$\frac{\partial}{\partial b_0}(u_i^2) = 2u_i \cdot (-1) = -2\left(y_i - (b_0 + b_1 x_i)\right)$$

Sum over all $n$ points and divide by $n$:

$$\frac{\partial J}{\partial b_0} = -\frac{2}{n}\sum_{i=1}^{n}\left(y_i - (b_0 + b_1 x_i)\right)$$

### Deriving $\frac{\partial J}{\partial b_1}$

Same setup, but differentiate the inner function with respect to $b_1$ instead:

$$\frac{\partial u_i}{\partial b_1} = -x_i$$

Combine:

$$\frac{\partial}{\partial b_1}(u_i^2) = 2u_i \cdot (-x_i) = -2x_i\left(y_i - (b_0 + b_1 x_i)\right)$$

Sum and divide by $n$:

$$\frac{\partial J}{\partial b_1} = -\frac{2}{n}\sum_{i=1}^{n} x_i\left(y_i - (b_0 + b_1 x_i)\right)$$

### Gradient Descent Update Rule

Using a learning rate $\alpha$, at each epoch:

$$b_0 \leftarrow b_0 - \alpha \frac{\partial J}{\partial b_0}$$

$$b_1 \leftarrow b_1 - \alpha \frac{\partial J}{\partial b_1}$$

### Implementation Notes

- Initial values: $b_0 = 0$, $b_1 = 0$
- Learning rate: $\alpha = 0.01$
- Epochs: 1000
- The error term $(y_i - \hat{y}_i)$ must be computed as actual minus predicted, matching the sign convention used throughout this derivation. Using the opposite convention without also flipping the sign in the update rule would send gradient descent in the wrong direction.

---

## Part 2: Logistic Regression

### Model

$$z_i = b_0 + b_1 x_i$$

$$\hat{y}_i = \sigma(z_i) = \frac{1}{1+e^{-z_i}}$$

### Cost Function

Cross-entropy loss over $n$ data points:

$$J(b_0, b_1) = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right]$$

### The Sigmoid Derivative

Write $\sigma(z) = (1+e^{-z})^{-1}$. Using the chain rule:

$$\sigma'(z) = -1 \cdot (1+e^{-z})^{-2} \cdot \frac{d}{dz}(1+e^{-z})$$

Since $\frac{d}{dz}(1+e^{-z}) = -e^{-z}$:

$$\sigma'(z) = \frac{e^{-z}}{(1+e^{-z})^2}$$

Rearranging $\sigma(z) = \frac{1}{1+e^{-z}}$ gives $e^{-z} = \frac{1}{\sigma(z)} - 1$. Substituting this in and simplifying:

$$\sigma'(z) = \sigma(z)(1-\sigma(z))$$

Since $\hat{y}_i = \sigma(z_i)$, this means:

$$\frac{\partial \hat{y}_i}{\partial z_i} = \hat{y}_i(1-\hat{y}_i)$$

### Deriving $\frac{\partial J}{\partial \hat{y}_i}$

Let $u_i = y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)$, so $J = -\frac{1}{n}\sum_{i=1}^n u_i$.

Differentiating $u_i$ with respect to $\hat{y}_i$:

$$\frac{\partial u_i}{\partial \hat{y}_i} = \frac{y_i}{\hat{y}_i} - \frac{1-y_i}{1-\hat{y}_i}$$

Since only the $i$-th term in the sum depends on $\hat{y}_i$, every other term's derivative with respect to $\hat{y}_i$ is zero, so:

$$\frac{\partial J}{\partial \hat{y}_i} = -\frac{1}{n}\left(\frac{y_i}{\hat{y}_i} - \frac{1-y_i}{1-\hat{y}_i}\right)$$

### Deriving $\frac{\partial z_i}{\partial b_0}$ and $\frac{\partial z_i}{\partial b_1}$

Since $z_i = b_0 + b_1 x_i$:

$$\frac{\partial z_i}{\partial b_0} = 1, \qquad \frac{\partial z_i}{\partial b_1} = x_i$$

### Combining the Chain Rule

$$\frac{\partial J}{\partial b_0} = \frac{\partial J}{\partial \hat{y}_i} \cdot \frac{\partial \hat{y}_i}{\partial z_i} \cdot \frac{\partial z_i}{\partial b_0}$$

$$\frac{\partial J}{\partial b_1} = \frac{\partial J}{\partial \hat{y}_i} \cdot \frac{\partial \hat{y}_i}{\partial z_i} \cdot \frac{\partial z_i}{\partial b_1}$$

Substituting all three pieces in and simplifying algebraically, the logarithmic and sigmoid terms cancel down to:

$$\frac{\partial J}{\partial b_0} = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)$$

$$\frac{\partial J}{\partial b_1} = \frac{1}{n}\sum_{i=1}^{n} x_i(\hat{y}_i - y_i)$$

### Gradient Descent Update Rule

$$b_0 \leftarrow b_0 - \alpha \frac{\partial J}{\partial b_0}$$

$$b_1 \leftarrow b_1 - \alpha \frac{\partial J}{\partial b_1}$$

### Implementation Notes

- Initial values: $b_0 = 0$, $b_1 = 0$
- Learning rate: $\alpha = 0.1$
- Epochs: 1000
- The error term here is $(\hat{y}_i - y_i)$, predicted minus actual, the opposite convention from linear regression. There is no factor of 2 in these gradients, unlike the linear regression case, since the 2 cancels out during the cross-entropy derivation.