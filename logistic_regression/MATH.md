# Logistic Regression

## Sigmoid Function

Maps any real number to (0, 1), turning a linear score into a probability:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

```python
# logistic_regression.py, lines 10-11
def _sigmoid(self, z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
```

> `np.clip(z, -500, 500)` prevents overflow in `np.exp` for extreme values of z.

Useful properties:
- $\sigma(0) = 0.5$
- $\sigma(-\infty) \to 0$, $\sigma(+\infty) \to 1$
- Derivative: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$

## Model

Given input features $\mathbf{x} \in \mathbb{R}^d$, weights $\mathbf{w} \in \mathbb{R}^d$, and bias $b$:

$$P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x} + b) = \hat{p}$$

```python
# logistic_regression.py, line 34
z = X @ self.w + self.b

# logistic_regression.py, line 35
p = self._sigmoid(z)
```

> `X @ self.w` computes the matrix-vector product $\mathbf{X}\mathbf{w}$ (dot product for all samples at once), then adds bias $b$. The sigmoid maps the result to probabilities.

The decision boundary is the hyperplane $\mathbf{w}^T\mathbf{x} + b = 0$. Predict class 1 if $\hat{p} \geq 0.5$, else class 0.

```python
# logistic_regression.py, line 61
return (self.predict_proba(X) >= 0.5).astype(int)
```

> Thresholding at 0.5 converts continuous probabilities into binary class predictions.

## Weight Initialization

Weights are initialized to zero and bias to zero:

$$\mathbf{w} = \mathbf{0}, \quad b = 0$$

```python
# logistic_regression.py, lines 30-32
m, n = X.shape
self.w = np.zeros(n)
self.b = 0
```

> `X.shape` gives (m_samples, n_features). Weights are a zero vector of length $n$.

## Binary Cross-Entropy Loss

For $n$ training examples with labels $y_i \in \{0, 1\}$:

$$L(\mathbf{w}, b) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

```python
# logistic_regression.py, lines 13-15
def _log_loss(self, y, p):
    eps = 1e-15
    return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
```

> `np.mean(...)` computes the $\frac{1}{n}\sum$ automatically. The `eps = 1e-15` prevents $\log(0)$ numerical instability.

- When $y_i = 1$: only $-\log(\hat{p}_i)$ matters — loss is 0 if $\hat{p} = 1$, explodes as $\hat{p} \to 0$.
- When $y_i = 0$: only $-\log(1 - \hat{p}_i)$ matters — loss is 0 if $\hat{p} = 0$, explodes as $\hat{p} \to 1$.

## Gradient

For a single example, using $\frac{\partial \hat{p}}{\partial z} = \hat{p}(1-\hat{p})$ where $z = \mathbf{w}^T\mathbf{x} + b$:

$$\frac{\partial L}{\partial z} = \hat{p} - y$$

$$\frac{\partial L}{\partial \mathbf{w}} = \frac{1}{n} \sum_{i=1}^{n} (\hat{p}_i - y_i)\mathbf{x}_i$$

$$\frac{\partial L}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} (\hat{p}_i - y_i)$$

```python
# logistic_regression.py, lines 36-38
dw = (1 / m) * (X.T @ (p - y))
db = (1 / m) * np.sum(p - y)
```

> `p - y` is the error vector $(\hat{p}_i - y_i)$. `X.T @ (p - y)` computes $\sum_i (\hat{p}_i - y_i)\mathbf{x}_i$ as a single matrix multiplication. Dividing by $m$ gives the mean gradient.

Gradient descent update:

$$\mathbf{w} \leftarrow \mathbf{w} - \alpha \frac{\partial L}{\partial \mathbf{w}}, \quad b \leftarrow b - \alpha \frac{\partial L}{\partial b}$$

```python
# logistic_regression.py, lines 39-40
self.w -= self.lr * dw
self.b -= self.lr * db
```

> `self.lr` is the learning rate $\alpha$. The loop at line 33 (`for _ in range(self.epochs)`) repeats this update for the configured number of iterations.

## Multi-Class (One-vs-Rest)

For $K$ classes, train $K$ binary classifiers. Each classifier $c$ learns $P(y = c \mid \mathbf{x})$:

$$\hat{p}_c = \sigma(\mathbf{w}_c^T\mathbf{x} + b_c)$$

```python
# logistic_regression.py, lines 42-48
def _fit_multi(self, X, y, n_classes):
    self.models_ = []
    for c in self.classes_:
        y_bin = (y == c).astype(float)
        m = LogisticRegression(lr=self.lr, epochs=self.epochs)
        m._fit_binary(X, y_bin)
        self.models_.append(m)
```

> For each class $c$, a binary label vector is created (`y == c` maps to 0/1), and a separate `LogisticRegression` model is trained via `_fit_binary`. The resulting models are stored in `self.models_`.

Final prediction uses normalization across all class probabilities:

$$P(y = c \mid \mathbf{x}) = \frac{\hat{p}_c}{\sum_{k=1}^{K} \hat{p}_k}$$

```python
# logistic_regression.py, lines 52-54
if hasattr(self, 'models_'):
    probs = np.column_stack([m._sigmoid(X @ m.w + m.b) for m in self.models_])
    return probs / probs.sum(axis=1, keepdims=True)
```

> Each model's sigmoid output is column-stacked into a (m × K) matrix. Dividing by the row-wise sum normalizes probabilities so they sum to 1.

Predicted class is the argmax:

$$\hat{y} = \arg\max_c P(y = c \mid \mathbf{x})$$

```python
# logistic_regression.py, line 60
return self.classes_[np.argmax(probs, axis=1)]
```

> `np.argmax(probs, axis=1)` finds the index of the highest probability per sample; indexing into `self.classes_` maps it back to the original class label.

## Why Log-Loss Works

Each example is a Bernoulli trial:

$$P(y_i \mid \mathbf{x}_i) = \hat{p}_i^{y_i} (1 - \hat{p}_i)^{1 - y_i}$$

Assuming independence, the likelihood of the entire dataset is:

$$\mathcal{L} = \prod_{i=1}^{n} \hat{p}_i^{y_i} (1 - \hat{p}_i)^{1 - y_i}$$

Taking the negative log-likelihood:

$$-\log \mathcal{L} = -\sum_{i=1}^{n} \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

This is exactly the binary cross-entropy loss (scaled by $n$). Minimizing log-loss is equivalent to maximizing the likelihood of the observed data — the core principle of Maximum Likelihood Estimation.

```python
# logistic_regression.py, line 15 (the same _log_loss)
return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
```

> The loss function directly implements the negative log-likelihood. `np.mean` normalizes by $n$, matching the $\frac{1}{n}$ scaling in the loss definition.
