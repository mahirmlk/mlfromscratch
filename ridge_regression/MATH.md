# Ridge Regression — Mathematical Explanation

## Why Regularization?

Ordinary Least Squares (OLS) minimizes the residual sum of squares. When features are
correlated or the number of features is large, OLS can **overfit** — it memorizes noise
in training data, producing large, unstable coefficient estimates that generalize poorly.

**Regularization** adds a penalty term that constrains model complexity, trading a small
increase in bias for a significant reduction in variance.

## The L2 Penalty

Ridge Regression adds an **L2 penalty** on the coefficient magnitudes to the OLS objective:

$$L(\mathbf{w}) = \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \alpha\|\mathbf{w}\|_2^2$$

where:
- $\mathbf{y} \in \mathbb{R}^n$ is the target vector
- $\mathbf{X} \in \mathbb{R}^{n \times d}$ is the design matrix
- $\mathbf{w} \in \mathbb{R}^d$ is the coefficient vector
- $\alpha \geq 0$ is the regularization strength (hyperparameter)

The first term fits the data; the second term penalizes large weights.

**Code:** `ridge_regression.py:5-6` — the regularization strength is stored as a hyperparameter:
```python
def __init__(self, alpha=1.0, solver='closed', lr=0.01, epochs=1000):
    self.alpha = alpha                                      # line 6: regularization strength α
```

**Code:** `ridge_regression.py:14` — the data dimensions $n$ and $d$ are extracted here:
```python
n, d = X.shape                                            # line 14: n = samples, d = features
```

## Closed-Form Solution

Taking the gradient and setting it to zero:

$$\frac{\partial L}{\partial \mathbf{w}} = -2\mathbf{X}^T(\mathbf{y} - \mathbf{X}\mathbf{w}) + 2\alpha\mathbf{w} = 0$$

Solving for $\mathbf{w}$:

$$\boxed{\mathbf{w} = (\mathbf{X}^T\mathbf{X} + \alpha \mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}}$$

**Code:** `ridge_regression.py:15-19` — the closed-form solver, step by step:

```python
if self.solver == 'closed':                               # line 15: select closed-form path
```

```python
    I = np.eye(d)                                         # line 17: build d×d identity matrix I
```
This is the $\mathbf{I}$ in $(\mathbf{X}^T\mathbf{X} + \alpha \mathbf{I})$.

```python
    self.w = np.linalg.solve(X.T @ X + self.alpha * I, X.T @ y)  # line 18: w = (X^T X + αI)^{-1} X^T y
```
This single line computes the entire closed-form solution. `X.T @ X` is $\mathbf{X}^T\mathbf{X}$, `self.alpha * I` is $\alpha\mathbf{I}$, and `X.T @ y` is $\mathbf{X}^T\mathbf{y}$. `np.linalg.solve(A, b)` solves $\mathbf{A}\mathbf{w} = \mathbf{b}$ without explicitly computing the inverse — numerically more stable and faster than `inv(A) @ b`.

```python
    self.b = np.mean(y - X @ self.w)                      # line 19: bias = mean of residuals
```
The bias term $b$ is not regularized (no intercept penalty). It's computed as the mean residual $\bar{y} - \bar{X}w$ after solving for $\mathbf{w}$.

### Why this is always solvable

OLS requires $(\mathbf{X}^T\mathbf{X})$ to be invertible. Ridge adds $\alpha\mathbf{I}$
to the diagonal, guaranteeing positive definiteness and thus invertibility — even when
$\mathbf{X}$ is rank-deficient or has collinear features.

**Code connection:** `np.linalg.solve` at line 18 will succeed even when `np.linalg.inv(X.T @ X)` (used in plain OLS) would fail on a near-singular matrix. The `self.alpha * I` term at line 18 is what makes the matrix well-conditioned — it shifts every eigenvalue up by $\alpha$.

## Gradient Descent Variant

When `solver != 'closed'`, the model is trained iteratively. The gradient of the Ridge loss is:

$$\nabla_w L = \frac{2}{n}X^T(X\mathbf{w} - \mathbf{y}) + \frac{2\alpha}{n}\mathbf{w}$$

$$\nabla_b L = \frac{2}{n}\sum_{i=1}^{n}(X\mathbf{w} + b - y_i)$$

**Code:** `ridge_regression.py:20-28` — the gradient descent solver, step by step:

```python
else:                                                     # line 20: gradient descent path
```

```python
    self.w = np.zeros(d)                                  # line 21: initialize w = 0 (d-dimensional)
    self.b = 0.0                                          # line 22: initialize bias = 0
```
Weights start at zero — no random initialization needed for this convex problem.

```python
    for _ in range(self.epochs):                          # line 23: iterate for self.epochs steps
```

```python
        y_pred = X @ self.w + self.b                      # line 24: ŷ = Xw + b (forward pass)
```
This is the prediction $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b$.

```python
        err = y_pred - y                                  # line 25: residual vector (ŷ - y)
```
This is $(\mathbf{X}\mathbf{w} + b - \mathbf{y})$, the residual needed for the gradient.

```python
        # grad_w = (2/n) X^T err + (2*alpha/n) w
        self.w -= self.lr * ((2.0 / n) * (X.T @ err) + (2.0 * self.alpha / n) * self.w)  # line 27
```
The weight update: $\mathbf{w} \leftarrow \mathbf{w} - \eta \left[\frac{2}{n}\mathbf{X}^T(\hat{\mathbf{y}} - \mathbf{y}) + \frac{2\alpha}{n}\mathbf{w}\right]$. The first term `(2/n) X.T @ err` is the OLS gradient; the second term `(2*alpha/n) * self.w` is the L2 penalty gradient that shrinks weights toward zero at every step.

```python
        self.b -= self.lr * (2.0 / n) * np.sum(err)      # line 28: bias update (no regularization)
```
The bias update: $b \leftarrow b - \eta \cdot \frac{2}{n}\sum(\hat{y}_i - y_i)$. Note that the bias is **not** regularized — only the weight vector $\mathbf{w}$ receives the L2 penalty.

### Convergence

The Ridge loss is strictly convex (the Hessian $\frac{2}{n}\mathbf{X}^T\mathbf{X} + \frac{2\alpha}{n}\mathbf{I}$ is positive definite), so gradient descent is guaranteed to converge to the global minimum. The learning rate `self.lr` (line 8) controls step size; `self.epochs` (line 9) controls iteration count.

## Prediction

$$\hat{y} = \mathbf{X}\mathbf{w} + b$$

**Code:** `ridge_regression.py:31-32` — the predict method:
```python
def predict(self, X):                                     # line 31
    return X @ self.w + self.b                            # line 32: ŷ = Xw + b
```
This is the same formula as the forward pass at line 24, but used for inference after training.

## What Does α Do?

| $\alpha$ | Behavior | Code effect |
|----------|----------|-------------|
| $\alpha = 0$ | Reduces to ordinary least squares | `self.alpha * I` = zero matrix at line 18; penalty term vanishes at line 27 |
| Small $\alpha$ | Mild shrinkage; close to OLS | Diagonal barely perturbed at line 18 |
| Large $\alpha$ | Strong shrinkage; coefficients pushed toward zero | Diagonal dominates at line 18, $w \approx X^Ty / \alpha$ |
| $\alpha \to \infty$ | All weights converge to zero | $w \to 0$ |

**Code:** `ridge_regression.py:5` — default `alpha=1.0`, a moderate regularization:
```python
def __init__(self, alpha=1.0, solver='closed', lr=0.01, epochs=1000):
```

Ridge **shrinks** coefficients toward zero but never sets them exactly to zero
(unlike L1 / Lasso). This makes it a good choice when many features contribute
small amounts, rather than a few features dominating.

## Geometric Intuition

Imagine the optimization in coefficient space $(w_1, w_2)$:

- The **OLS objective** $\|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2$ forms concentric
  ellipses (contours of equal loss).
- The **L2 constraint** $\|\mathbf{w}\|_2^2 \leq t$ defines a **circle** (disk) centered
  at the origin.

Ridge regression finds the point where an ellipse first touches the circle. Because
the constraint region is smooth (no corners), the solution almost never lands on an
axis — coefficients are shrunk but not eliminated. This contrasts with Lasso (L1),
whose diamond-shaped constraint region encourages sparsity.

```
        w₂
        |      ╱ ellipse contours
        |    ╱
   ─────┼──●───── w₁     ● = Ridge solution (circle ∩ ellipse)
        |    ╲
        |      ╲
```

## Summary

Ridge regression is OLS with a budget on coefficient magnitude. It prevents overfitting,
handles multicollinearity, and has an elegant closed-form solution. The single
hyperparameter $\alpha$ controls the bias–variance tradeoff.

| Concept | Code location |
|---------|--------------|
| Regularization strength $\alpha$ | `ridge_regression.py:5-6` |
| Data shape extraction | `ridge_regression.py:14` |
| Closed-form: identity matrix $\mathbf{I}$ | `ridge_regression.py:17` |
| Closed-form: $\mathbf{w} = (X^TX + \alpha I)^{-1}X^Ty$ | `ridge_regression.py:18` |
| Closed-form: bias from residuals | `ridge_regression.py:19` |
| GD: weight initialization | `ridge_regression.py:21` |
| GD: forward pass $\hat{y} = Xw + b$ | `ridge_regression.py:24` |
| GD: residual computation | `ridge_regression.py:25` |
| GD: weight update with L2 penalty | `ridge_regression.py:27` |
| GD: bias update (unregularized) | `ridge_regression.py:28` |
| Prediction | `ridge_regression.py:32` |

See `../linear_regression/adversarial_demo.py` for a demonstration of OLS failing on multicollinear data and Ridge fixing it.
