# Support Vector Machine

## Geometric Margin

The margin is the distance between the two parallel supporting hyperplanes
closest to the decision boundary. For a linear classifier $f(x) = w^Tx + b$,
the functional margin of a point is $y_i(w^Tx_i + b)$ and the geometric
margin is:

$$\gamma = \frac{2}{||w||}$$

SVM maximizes $\gamma$, which is equivalent to minimizing $||w||$.

---

## Hard-Margin SVM

When data is linearly separable with no errors allowed:

$$\min_{w,b} \; \frac{1}{2}||w||^2$$
$$\text{s.t.} \quad y_i(w^Tx_i + b) \geq 1, \quad \forall\, i = 1,\dots,N$$

This is a convex quadratic program. The constraint ensures every point is
correctly classified with functional margin $\geq 1$.

---

## Soft-Margin SVM

Real data is noisy. Slack variables $\xi_i \geq 0$ allow margin violations:

$$\min_{w,b,\xi} \; \frac{1}{2}||w||^2 + C\sum_{i=1}^{N}\xi_i$$
$$\text{s.t.} \quad y_i(w^Tx_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

- $C > 0$ trades off margin size vs. training error.
- Small $C$ → wider margin, more tolerance for misclassification.
- Large $C$ → narrower margin, stricter classification.

The hyperparameter $C$ is set during initialization:

```python
# svm.py — line 6
self.C = C
```

---

## Hinge Loss

The soft-margin objective is equivalent to the unconstrained regularized
empirical risk:

$$L = \frac{1}{2}||w||^2 + C\sum_{i=1}^{N} \max\!\big(0,\; 1 - y_i(w^Tx_i + b)\big)$$

The per-sample loss is the **hinge loss**:

$$\ell_i = \max(0,\; 1 - y_i(w^Tx_i + b))$$

- Zero when the point is correctly classified with margin $\geq 1$.
- Linear penalty otherwise.

The functional margin $y_i(w^Tx_i + b)$ is computed per sample:

```python
# svm.py — line 19
margin = y[i] * (np.dot(X[i], self.w) + self.b)
```

The hinge condition $\ell_i > 0$ corresponds to `margin < 1`:

```python
# svm.py — line 20
if margin < 1:
```

---

## SGD Update for Hinge Loss

For stochastic gradient descent, sample one point $(x_i, y_i)$:

1. If $y_i(w^Tx_i + b) \geq 1$ (correct side, outside margin):
$$w \leftarrow w - \eta\,\lambda\, w, \quad b \leftarrow b$$

```python
# svm.py — line 24
self.w -= self.lr * self.w / self.C
```

Only the weight vector is regularized; bias remains unchanged (no update).

2. If $y_i(w^Tx_i + b) < 1$ (inside margin or misclassified):
$$w \leftarrow w - \eta\,(\lambda\, w - y_i\, x_i)$$
$$b \leftarrow b + \eta\, y_i$$

```python
# svm.py — lines 21-22
self.w -= self.lr * (self.w / self.C - y[i] * X[i])
self.b -= self.lr * (-y[i])
```

where $\eta$ is the learning rate and $\lambda = \frac{1}{C}$ is the
regularization coefficient. Note `self.w / self.C` is $\lambda w$, and
`-y[i] * X[i]` is $-y_i x_i$, so the subtraction produces the gradient
step $w - \eta(\lambda w - y_i x_i)$.

The learning rate and epochs are set here:

```python
# svm.py — lines 7-8
self.lr = lr
self.epochs = epochs
```

Weights and bias are initialized to zero:

```python
# svm.py — lines 14-15
self.w = np.zeros(d)
self.b = 0.0
```

---

## Support Vectors

**Support vectors** are the training points that lie exactly on the margin
boundaries, i.e. those with $y_i(w^Tx_i + b) = 1$ (or $\xi_i > 0$ in the
soft-margin case). They are the only points that influence the decision
boundary — removing any non-support-vector point leaves the solution
unchanged. This is the essence of SVM's sparsity.

In this implementation, support vectors are implicit — they are the points
that trigger the `margin < 1` branch during training.

---

## Prediction

The decision function $f(x) = \text{sign}(w^Tx + b)$:

```python
# svm.py — line 27
return np.sign(np.dot(X, self.w) + self.b)
```

Accuracy is the fraction of correct predictions:

```python
# svm.py — line 30
return np.mean(self.predict(X) == y)
```

---

## Kernel Trick (Optional Extension)

To handle non-linear boundaries, replace inner products with a kernel
$K(x_i, x_j) = \phi(x_i)^T\phi(x_j)$, implicitly mapping data to a
higher-dimensional feature space without computing $\phi$ directly. Common
choices: polynomial, RBF (Gaussian), sigmoid.

This implementation uses the linear kernel (standard dot product) only.
