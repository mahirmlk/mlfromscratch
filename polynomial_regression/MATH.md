# Polynomial Regression — Math

## Core Idea

Polynomial regression fits a nonlinear curve by mapping inputs into a
higher-dimensional feature space, then applying **ordinary linear regression**
in that space.

## Class Initialisation

The model stores the polynomial degree and initialises weights to `None`:

```python
# polynomial_regression.py lines 6-8
def __init__(self, degree=2):
    self.degree = degree
    self.coef = None
```

> `self.coef` will hold the weight vector $\mathbf{w}$ after fitting.

## Feature Expansion

For a single input $x$ and polynomial degree $d$, define the basis function:

$$\phi(x) = [1,\; x,\; x^2,\; \ldots,\; x^d]$$

The prediction becomes a linear model in the expanded space:

$$\hat{y} = \mathbf{w}^\top \phi(x) = w_0 + w_1 x + w_2 x^2 + \cdots + w_d x^d$$

### Example: degree 2 (one feature)

$$\hat{y} = w_0 + w_1 x + w_2 x^2$$

The design matrix $\\Phi$ has rows $\\phi(x_i)$; we solve the normal equation
exactly as in linear regression:

$$\mathbf{w}^* = (\Phi^\top \Phi)^{-1}\, \Phi^\top \mathbf{y}$$

### `_expand` — Building the Design Matrix $\Phi$

The `_expand` method transforms raw input $X$ into the polynomial feature matrix $\Phi$:

```python
# polynomial_regression.py lines 10-17
def _expand(self, X):
    n, d = X.shape
    features = [np.ones(n)]
    for deg in range(1, self.degree + 1):
        for combo in combinations_with_replacement(range(d), deg):
            feat = np.prod(X[:, combo], axis=1)
            features.append(feat)
    return np.column_stack(features)
```

**Line-by-line:**

| Lines | Math | Code |
|---|---|---|
| 11 | $n$ samples, $d$ input dimensions | `n, d = X.shape` |
| 12 | Bias column $\phi_0 = 1$ for every sample | `features = [np.ones(n)]` |
| 13 | Iterate over polynomial degrees $1 \ldots d$ | `for deg in range(1, self.degree + 1)` |
| 14 | Generate all monomials of that degree (with repetition) | `combinations_with_replacement(range(d), deg)` |
| 15 | Each monomial = product of selected features: $\prod_{j \in \text{combo}} X_j$ | `np.prod(X[:, combo], axis=1)` |
| 16 | Append the monomial column to $\Phi$ | `features.append(feat)` |
| 17 | Stack all columns into the full design matrix $\Phi$ | `np.column_stack(features)` |

> For degree $d$ in $p$ dimensions this produces $\binom{p+d}{d}$ expanded features.

## Multivariate Extension

For $p$ input features $[x_1, x_2, \ldots, x_p]$, degree $d$ includes all
monomials up to order $d$, e.g. for $d=2$:

$$\phi(\mathbf{x}) = [1,\; x_1,\; x_2,\; \ldots,\; x_p,\;
x_1^2,\; x_2^2,\; \ldots,\; x_p^2,\;
x_1 x_2,\; x_1 x_3,\; \ldots]$$

The number of expanded features for degree $d$ in $p$ dimensions is
$\binom{p+d}{d}$, which grows rapidly.

## Fitting — Closed-Form Solution

The `fit` method builds $\Phi$ then solves the normal equation via least squares:

```python
# polynomial_regression.py lines 19-23
def fit(self, X, y):
    Phi = self._expand(X)
    # normal equation: w = (Phi^T Phi)^-1 Phi^T y
    self.coef = np.linalg.lstsq(Phi, y, rcond=None)[0]
    return self
```

**Line-by-line:**

| Lines | Math | Code |
|---|---|---|
| 20 | Build design matrix $\Phi = \phi(X)$ | `Phi = self._expand(X)` |
| 22 | Solve $\mathbf{w}^* = (\Phi^\top \Phi)^{-1}\, \Phi^\top \mathbf{y}$ | `np.linalg.lstsq(Phi, y, rcond=None)[0]` |

> `lstsq` numerically solves the least-squares problem, which is mathematically
> equivalent to the normal equation but more numerically stable than explicitly
> inverting $\Phi^\top \Phi$.

## Prediction

Prediction is a simple matrix–vector product $\hat{\mathbf{y}} = \Phi \mathbf{w}$:

```python
# polynomial_regression.py lines 25-26
def predict(self, X):
    return self._expand(X) @ self.coef
```

> Expand $X$ into $\Phi$, then multiply by weights: exactly the linear model
> $\hat{\mathbf{y}} = \Phi\, \mathbf{w}$.

## Overfitting Risk

- Higher $d$ → more parameters → lower training error but higher variance.
- A degree-$d$ polynomial can perfectly fit $d+1$ points (noise and all).
- Mitigation: cross-validation to select $d$, or add L2 regularisation
  (Ridge polynomial regression):

$$\mathbf{w}^* = (\Phi^\top \Phi + \lambda I)^{-1}\, \Phi^\top \mathbf{y}$$

> Note: the implementation uses unregularised `lstsq`; the Ridge variant
> above is the mathematical extension, not yet implemented in this class.

## Summary

| Aspect | Detail | Code |
|---|---|---|
| Model | $\hat{y} = \sum_{j=0}^{d} w_j\, x^j$ | `_expand` + `@ self.coef` (lines 25-26) |
| Design matrix | $\Phi$ with all monomials up to degree $d$ | `_expand` (lines 10-17) |
| Loss | MSE: $\frac{1}{n}\sum_i (y_i - \hat{y}_i)^2$ | — |
| Solution | Closed-form normal equation on $\Phi$ | `np.linalg.lstsq` (line 22) |
| Complexity | $\mathcal{O}(nd^2)$ for $n$ samples, degree $d$ | — |
