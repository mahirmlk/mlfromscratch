# Elastic Net — Mathematical Foundations

## Objective Function

Elastic Net combines L1 (Lasso) and L2 (Ridge) penalties into a single regularized loss:

$$L(\mathbf{w}) = \|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2_2 + \alpha\left(\lambda\|\mathbf{w}\|_1 + \frac{1-\lambda}{2}\|\mathbf{w}\|_2^2\right)$$

where:
- $\mathbf{y} \in \mathbb{R}^n$ is the target vector
- $\mathbf{X} \in \mathbb{R}^{n \times p}$ is the design matrix
- $\mathbf{w} \in \mathbb{R}^p$ is the coefficient vector
- $\alpha > 0$ controls overall regularization strength
- $\lambda \in [0, 1]$ (often called `l1_ratio`) balances L1 vs L2

**In code:** The hyperparameters $\alpha$ and $\lambda$ (`l1_ratio`) are stored on init, and decomposed into pure L1/L2 strengths during `fit`:

```python
# elastic_net.py lines 5-9: hyperparameter storage
def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4):
    self.alpha = alpha
    self.l1_ratio = l1_ratio
    self.max_iter = max_iter
    self.tol = tol
```

```python
# elastic_net.py lines 18-19: decompose alpha into L1 and L2 strengths
l1 = self.alpha * self.l1_ratio        # α·λ  (L1 penalty weight)
l2 = self.alpha * (1 - self.l1_ratio)  # α·(1-λ) (L2 penalty weight)
```

`l1` corresponds to $\alpha\lambda$ and `l2` corresponds to $\alpha(1-\lambda)$ in the formula above.

## The `l1_ratio` Parameter

| `l1_ratio` | Effect |
|------------|--------|
| 1.0 | Pure Lasso (L1 only) |
| 0.0 | Pure Ridge (L2 only) |
| 0 < r < 1 | Elastic Net blend |

The `l1_ratio` $\lambda$ interpolates smoothly between the two penalties. In practice, values around 0.5–0.7 are common starting points.

**In code:** When `l1_ratio=1.0`, `l2` becomes `0` (pure Lasso). When `l1_ratio=0.0`, `l1` becomes `0` (pure Ridge). See lines 18–19 above.

## When to Use Elastic Net vs Lasso vs Ridge

- **Ridge** ($\lambda = 0$): Best when many features contribute small effects. Shrinks coefficients but never sets them to zero.
- **Lasso** ($\lambda = 1$): Best for sparse models. Can zero out coefficients, performing feature selection. Struggles when features are correlated (picks one arbitrarily).
- **Elastic Net** ($0 < \lambda < 1$): Ideal when features are correlated and you want sparsity. Groups of correlated features are selected or dropped together, avoiding Lasso's arbitrary selection problem.

## Coordinate Descent Updates

The Elastic Net is commonly solved via coordinate descent. For each coefficient $w_j$, hold all others fixed and minimize the partial objective.

### Initialization

Before iterating, initialize the coefficient vector and compute an initial residual:

$$\mathbf{w} = \mathbf{0}, \quad b = \bar{y}, \quad r = \mathbf{y} - b$$

**In code:**

```python
# elastic_net.py lines 12-15: initialization
n, p = X.shape
self.w = np.zeros(p)          # w = 0
self.b = np.mean(y)            # b = mean(y), initial intercept
r = y - self.b                 # residual = y - b
```

```python
# elastic_net.py line 17: precompute column norms for denominator
col_sq = np.sum(X ** 2, axis=0) / n   # ‖xⱼ‖² / n for each feature
```

`col_sq[j]` stores $\mathbf{x}_j^\top \mathbf{x}_j / n$, which appears in the denominator of the update rule.

### Partial Residual

Define the partial residual excluding feature $j$:

$$r^{(j)} = \mathbf{y} - \sum_{k \neq j} x_k w_k$$

**In code:** Rather than recomputing $r^{(j)}$ from scratch, the algorithm efficiently adds back the contribution of feature $j$ to the current residual:

```python
# elastic_net.py line 25: add back feature j's contribution to get r^(j)
r += X[:, j] * self.w[j]
```

This transforms $r = y - \sum_k x_k w_k$ into $r^{(j)} = r + x_j w_j = y - \sum_{k \neq j} x_k w_k$.

### Correlation and Soft-Thresholding Update

The update for $w_j$ uses the soft-thresholding operator $\mathcal{S}$:

$$w_j \leftarrow \frac{\mathcal{S}\!\left(\mathbf{x}_j^\top r^{(j)} / n,\, \alpha\lambda\right)}{\mathbf{x}_j^\top \mathbf{x}_j / n + \alpha(1-\lambda)}$$

where the soft-thresholding operator is:

$$\mathcal{S}(z, \gamma) = \text{sign}(z)\max(|z| - \gamma,\, 0)$$

**In code:**

```python
# elastic_net.py line 27: compute correlation ρ = xⱼᵀr^(j) / n
rho = np.dot(X[:, j], r) / n
```

`rho` is $\rho = \mathbf{x}_j^\top r^{(j)} / n$, the (unnormalized) correlation between feature $j$ and the partial residual.

```python
# elastic_net.py line 28: denominator = ‖xⱼ‖²/n + α(1-λ)  (L2 shrinkage)
denom = col_sq[j] + l2
```

`denom` is $\mathbf{x}_j^\top \mathbf{x}_j / n + \alpha(1-\lambda)$. The `l2` term ($\alpha(1-\lambda)$) provides the Ridge shrinkage in the denominator.

```python
# elastic_net.py line 29: apply soft-thresholding (L1) then divide by denom (L2)
self.w[j] = self._soft(rho, l1) / denom
```

This is the full update: $\mathcal{S}(\rho, \alpha\lambda) / \text{denom}$. Soft-thresholding (`l1` = $\alpha\lambda$) drives small coefficients to exactly zero (Lasso effect), while the denominator provides Ridge shrinkage.

```python
# elastic_net.py lines 42-43: soft-thresholding operator S(z, γ) = sign(z)·max(|z|-γ, 0)
def _soft(self, x, t):
    return np.sign(x) * np.maximum(np.abs(x) - t, 0)
```

The `_soft` method implements $\mathcal{S}(z, \gamma) = \text{sign}(z)\max(|z| - \gamma, 0)$.

### Update the Residual

After computing the new $w_j$, subtract its contribution back from the residual:

```python
# elastic_net.py line 30: subtract new wⱼ's contribution from residual
r -= X[:, j] * self.w[j]
```

This restores $r = y - \sum_k x_k w_k$ with the updated $w_j$.

### Intercept Update

After each full pass over all $p$ features, update the intercept:

$$b = \bar{r}$$

**In code:**

```python
# elastic_net.py line 33: update intercept as mean of current residual
self.b = np.mean(r)
```

Since $\mathbf{X}$ is implicitly centered through the residual updates, the optimal intercept is the mean of the residual.

### Convergence Check

After each outer iteration, check if coefficients have converged:

$$\max_j |w_j^{\text{new}} - w_j^{\text{old}}| < \texttt{tol}$$

**In code:**

```python
# elastic_net.py lines 22, 35-36: convergence check
w_old = self.w.copy()                                    # save previous weights
# ...
if np.max(np.abs(self.w - w_old)) < self.tol:            # max |Δw| < tol?
    break
```

If the largest coefficient change is below `tol`, the algorithm has converged and stops early.

## Full Algorithm Summary

1. **Initialize** $\mathbf{w} = \mathbf{0}$, $b = \bar{y}$, precompute column norms
2. **Repeat** until convergence (max `max_iter` times):
   - Save $\mathbf{w}_{\text{old}}$
   - For $j = 1, 2, \ldots, p$:
     - Add back $x_j w_j$ to residual → $r^{(j)}$
     - Compute $\rho = x_j^\top r^{(j)} / n$
     - Update $w_j = \mathcal{S}(\rho, \alpha\lambda) / (\|x_j\|^2/n + \alpha(1-\lambda))$
     - Subtract $x_j w_j$ from residual
   - Update intercept $b = \bar{r}$
   - Check convergence: $\|\mathbf{w} - \mathbf{w}_{\text{old}}\|_\infty < \texttt{tol}$
3. **Return** $\mathbf{w}, b$

## Prediction

Given learned weights and bias:

$$\hat{y} = \mathbf{X}\mathbf{w} + b$$

**In code:**

```python
# elastic_net.py lines 39-40: prediction
def predict(self, X):
    return X @ self.w + self.b
```

A simple matrix-vector product plus the intercept term.

## Convergence Properties

Coordinate descent converges because each step decreases the objective and the Elastic Net objective is convex (though not strictly convex at the L1 term). The `tol` parameter (default $10^{-4}$) and `max_iter` parameter (default 1000) control when the algorithm stops.
