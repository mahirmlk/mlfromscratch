# Lasso Regression

## 1. Objective Function

Lasso (Least Absolute Shrinkage and Selection Operator) minimizes:

$$L(\mathbf{w}) = \frac{1}{2}\|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \alpha\|\mathbf{w}\|_1$$

where $\alpha \geq 0$ controls regularization strength and $\|\mathbf{w}\|_1 = \sum_{j=1}^{p} |w_j|$.

Equivalently, the constrained form is:

$$\min_{\mathbf{w}} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 \quad \text{s.t.} \quad \|\mathbf{w}\|_1 \leq t$$

**Code — hyperparameters controlling regularization:**

```python
# lasso_regression.py, lines 5-8
def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
    self.alpha = alpha          # α — L1 regularization strength
    self.max_iter = max_iter    # maximum coordinate descent sweeps
    self.tol = tol              # convergence threshold for w change
```

`alpha` is the $\alpha$ in the Lasso objective. `max_iter` and `tol` control the iterative solver (coordinate descent has no closed-form solution).

---

## 2. Why L1 Produces Sparsity

The L1 constraint region $\|\mathbf{w}\|_1 \leq t$ forms a **diamond** (rotated hypercube) in parameter space. The OLS contours (ellipsoids of constant RSS) are smooth and convex.

Key geometric insight: the diamond has **sharp corners on the axes**. As the constraint shrinks (larger $\alpha$), the ellipsoid most likely touches the constraint boundary at a **corner**, where one or more $w_j = 0$. This is sparsity — automatic feature selection.

In contrast, Ridge's L2 constraint region is a **sphere** (smooth everywhere), so the tangency point almost never lands on an axis.

**Sparsity intuition:** The L1 norm grows linearly with $|w_j|$, making small nonzero values "expensive." The optimizer prefers to set coefficients to exactly zero rather than keep tiny values.

---

## 3. Initialization

$$\mathbf{w} = \mathbf{0}, \quad b = \bar{y}$$

Weights start at zero; the bias is initialized to the mean of $y$.

**Code — initialization:**

```python
# lasso_regression.py, lines 13-15
n, d = X.shape
self.w = np.zeros(d)         # w = 0 vector (length d)
self.b = np.mean(y)          # b = ȳ (mean of target)
```

The intercept is set to the sample mean of $y$ and held fixed during coordinate descent (standard practice — the data is assumed centered or the bias absorbs the mean).

---

## 4. Precomputing Column Norms

For the soft-thresholding update we need $\|\mathbf{x}_j\|_2^2$ for every feature $j$. Precomputing avoids redundant work inside the inner loop:

$$\text{col\_sq}_j = \|\mathbf{x}_j\|_2^2 = \sum_{i=1}^{n} x_{ij}^2$$

**Code — column norms squared:**

```python
# lasso_regression.py, lines 18-19
col_sq = np.sum(X ** 2, axis=0)           # ‖x_j‖² for each j
col_sq[col_sq == 0] = 1e-12               # guard against division by zero
```

---

## 5. Soft Thresholding (Coordinate Descent Solution)

For a single coordinate $w_j$ with all other weights fixed, the Lasso objective reduces to a 1D problem with closed-form solution:

$$w_j \leftarrow S\!\left(\frac{\mathbf{x}_j^\top(\mathbf{y} - \mathbf{X}_{-j}\mathbf{w}_{-j})}{\|\mathbf{x}_j\|_2^2},\; \frac{\alpha}{\|\mathbf{x}_j\|_2^2}\right)$$

where $S$ is the **soft thresholding operator**:

$$S(z, \gamma) = \operatorname{sign}(z)\max(|z| - \gamma,\; 0)$$

- If $|z| \leq \gamma$: the coefficient is shrunk **exactly to zero**.
- If $|z| > \gamma$: the coefficient is shrunk toward zero by amount $\gamma$.

This is why Lasso performs variable selection — it sets entire coefficients to zero.

**Code — partial residual $r_j = \mathbf{y} - \mathbf{X}_{-j}\mathbf{w}_{-j}$:**

```python
# lasso_regression.py, line 25
r_j = y - self.b - X @ self.w + X[:, j] * self.w[j]
#     ─────────────  ────────    ─────────────────────
#     full residual  all terms   re-add feature j's
#     with bias      Xw          contribution (undo j)
```

This computes $r_j = y - b - Xw + x_j w_j$, which is exactly $\mathbf{y} - \mathbf{X}_{-j}\mathbf{w}_{-j}$ (the residual excluding feature $j$).

**Code — $\rho_j = \mathbf{x}_j^\top r_j$:**

```python
# lasso_regression.py, line 26
rho_j = X[:, j] @ r_j          # dot product: x_j^T · r_j
```

**Code — soft-thresholding update:**

```python
# lasso_regression.py, line 28
self.w[j] = np.sign(rho_j) * max(abs(rho_j) - self.alpha * n / 2, 0) / col_sq[j]
#           ──────────────   ──────────────────────────────────────   ──────────
#           sign(ρ_j)        max(|ρ_j| - α·n/2, 0)                  1/‖x_j‖²
```

This is the soft-thresholding formula $S\!\left(\frac{\rho_j}{\|\mathbf{x}_j\|^2},\; \frac{\alpha}{\|\mathbf{x}_j\|^2}\right)$ implemented directly. The factor of $n/2$ appears because the code uses the average-loss formulation $\frac{1}{2n}\|\cdot\|^2$ rather than $\frac{1}{2}\|\cdot\|^2$, absorbing $n$ into $\alpha$.

---

## 6. Coordinate Descent Algorithm

```
Initialize w = 0 (or warm start)
Repeat until convergence:
    For j = 1, ..., p:
        r_j = y - X w + x_j w_j          # partial residual
        z_j = x_j^T r_j / n               # univariate OLS coefficient
        w_j = S(z_j, alpha / n)            # soft-threshold
```

**Code — full convergence loop:**

```python
# lasso_regression.py, lines 21-31
for _ in range(self.max_iter):                 # outer loop: sweeps
    w_old = self.w.copy()                      # snapshot for convergence check
    for j in range(d):                         # inner loop: each coordinate
        r_j = y - self.b - X @ self.w + X[:, j] * self.w[j]   # partial residual
        rho_j = X[:, j] @ r_j                                  # ρ_j
        self.w[j] = np.sign(rho_j) * max(abs(rho_j) - self.alpha * n / 2, 0) / col_sq[j]

    if np.max(np.abs(self.w - w_old)) < self.tol:   # ‖w_new - w_old‖∞ < tol
        break                                         # converged → stop early
```

**Complexity:** Each sweep costs $O(np)$. Convergence is guaranteed because the Lasso objective is convex and each coordinate update is exact.

---

## 7. Prediction

$$\hat{y} = \mathbf{X}\mathbf{w} + b$$

**Code — predict:**

```python
# lasso_regression.py, line 36
return X @ self.w + self.b     # ŷ = Xw + b
```

---

## 8. Lasso vs. Ridge

| Property | Lasso (L1) | Ridge (L2) |
|---|---|---|
| Penalty | $\alpha\sum\|w_j\|$ | $\alpha\sum w_j^2$ |
| Constraint shape | Diamond | Sphere |
| Sparsity | Yes — automatic feature selection | No — all coefficients nonzero |
| Closed-form | No (iterative) | Yes: $\mathbf{w} = (\mathbf{X}^\top\mathbf{X} + \alpha\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$ |
| Multicollinearity | Picks one predictor, zeros others | Splits weight among correlated predictors |
| Best for | Sparse models, high-dimensional $p \gg n$ | Dense models, correlated features |

---

## 9. Regularization Path

As $\alpha$ increases from 0:

- $\alpha = 0$: OLS solution (no shrinkage).
- Small $\alpha$: most coefficients nonzero, slightly shrunk.
- Large $\alpha$: most coefficients exactly zero.
- $\alpha \geq \alpha_{\max} = \max_j |\mathbf{x}_j^\top \mathbf{y}| / n$: all coefficients zero.

The **Elastic Net** combines L1 and L2: $\alpha\rho\|\mathbf{w}\|_1 + \frac{\alpha(1-\rho)}{2}\|\mathbf{w}\|_2^2$, gaining sparsity from L1 and stability from L2.
