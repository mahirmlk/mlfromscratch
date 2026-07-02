# XGBoost — Mathematical Derivation with Code References

## Regularized Objective

XGBoost minimizes a regularized loss function that balances training accuracy with model complexity:

$$L(\phi) = \sum_{i=1}^{n} l(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)$$

where $l$ is a differentiable convex loss (e.g., squared error, log-loss), $f_k$ is the $k$-th tree, and $\Omega$ penalizes tree complexity.

**Code:** The ensemble prediction adds each tree's contribution with a learning-rate weight. At fit time, the base prediction is the mean of `y` (squared-error loss).

```python
# xgboost_model.py lines 86–97 (fit loop)
self.base_pred = np.full(n, np.mean(y))       # line 88
pred = self.base_pred.copy()                    # line 89

for i in range(self.n_estimators):              # line 91
    g = pred - y                                # line 92
    h = np.ones(n)                              # line 93
    tree = XGBoostTree(...)                     # line 94
    tree.fit(X, g, h)                           # line 95
    pred += self.lr * tree.predict(X)           # line 96
```

> `g = pred - y` computes the gradient of squared-error loss (see §Gradient Statistics below). `h = np.ones(n)` is the constant Hessian for squared-error. Each round adds `lr * tree.predict(X)` — the additive training step.

---

## Tree Complexity Penalty

Each tree's complexity is measured by its number of leaves and the L2 norm of leaf weights:

$$\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$$

- $T$ — number of leaves
- $w_j$ — weight (score) of leaf $j$
- $\gamma$ — penalty per leaf (controls pruning)
- $\lambda$ — L2 regularization on weights

**Code:** The regularization hyperparameters are stored on the tree object and used in the gain formula (see §Split Gain).

```python
# xgboost_model.py lines 15–18
def __init__(self, max_depth=3, reg_lambda=1.0, gamma=0.0):
    self.reg_lambda = reg_lambda                # line 17 — λ
    self.gamma = gamma                          # line 18 — γ
```

> `reg_lambda` is $\lambda$ (L2 regularization on leaf weights). `gamma` is $\gamma$ (per-leaf complexity penalty). Both appear directly in the split-gain and leaf-weight formulas.

---

## Additive Training with Taylor Expansion

At boosting round $t$, we add tree $f_t$ to the ensemble:

$$\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + f_t(x_i)$$

The objective becomes:

$$L^{(t)} = \sum_{i=1}^{n} l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)$$

Applying a **second-order Taylor expansion** of $l$ around $\hat{y}_i^{(t-1)}$:

$$L^{(t)} \approx \sum_{i=1}^{n} \left[ g_i \, f_t(x_i) + \frac{1}{2} h_i \, f_t^2(x_i) \right] + \Omega(f_t)$$

**Code:** The additive update `pred += lr * tree.predict(X)` implements $\hat{y}^{(t)} = \hat{y}^{(t-1)} + \eta \, f_t(x)$ where $\eta$ is the learning rate.

```python
# xgboost_model.py line 96 (additive update)
pred += self.lr * tree.predict(X)

# xgboost_model.py lines 99–104 (predict — same pattern)
pred = np.full(n, self.base_pred.mean())        # line 101
for tree in self.trees:                          # line 102
    pred += self.lr * tree.predict(X)            # line 103
```

> Both training and inference accumulate `lr * tree.predict(X)` into `pred`. This is the core additive-training loop.

---

## Gradient Statistics

where the **gradient statistics** are:

$$g_i = \frac{\partial \, l(y_i, \hat{y}_i^{(t-1)})}{\partial \, \hat{y}_i^{(t-1)}}, \qquad h_i = \frac{\partial^2 \, l(y_i, \hat{y}_i^{(t-1)})}{\partial \, (\hat{y}_i^{(t-1)})^2}$$

**Code:** For squared-error loss $l = \frac{1}{2}(y - \hat{y})^2$, the first derivative is $g_i = \hat{y}_i - y_i$ and the second derivative is $h_i = 1$.

```python
# xgboost_model.py lines 92–93
g = pred - y                                    # line 92 — gradient: ∂l/∂ŷ = ŷ - y
h = np.ones(n)                                  # line 93 — Hessian:  ∂²l/∂ŷ² = 1
```

> This is a simplification for squared-error loss. A general XGBoost would support arbitrary loss functions with non-trivial Hessians.

---

## Aggregate Statistics per Leaf

For leaf $j$, define the summed gradient and Hessian over all instances that land in that leaf:

$$G_j = \sum_{i \in I_j} g_i, \qquad H_j = \sum_{i \in I_j} h_i$$

where $I_j = \{i \mid q(x_i) = j\}$ and $q$ is the tree's leaf-assignment function.

**Code:** At each node, the algorithm sums `g` and `h` over the samples that reach that node (the full set), and also over the left/right partitions after a candidate split.

```python
# xgboost_model.py line 26 — full-node aggregates (G, H)
G, H = np.sum(g), np.sum(h)

# xgboost_model.py lines 44–45 — left/right child aggregates
Gl, Hl = np.sum(g[mask]), np.sum(h[mask])       # line 44 — left child
Gr, Hr = G - Gl, H - Hl                         # line 45 — right child (complement)
```

> `G, H` are the aggregate statistics over all samples at the current node. `Gl, Hl` and `Gr, Hr` are the partitioned sums for left and right children. The right aggregates are computed as complements to avoid redundant summation.

---

## Optimal Leaf Weight

Substituting the aggregated statistics into the approximated objective and solving $\partial L / \partial w_j = 0$:

$$w_j^* = -\frac{G_j}{H_j + \lambda}$$

The corresponding optimal objective value for a given tree structure is:

$$L^* = -\frac{1}{2} \sum_{j=1}^{T} \frac{G_j^2}{H_j + \lambda} + \gamma T$$

**Code:** Every leaf node stores this optimal weight.

```python
# xgboost_model.py line 27
node.weight = -G / (H + self.reg_lambda)
```

> This is exactly $w^* = -G_j / (H_j + \lambda)$. The weight is computed for every node (including internal ones) before deciding whether to split further. If no beneficial split is found, the node becomes a leaf with this weight.

---

## Split Gain

To decide whether to split a leaf into left ($L$) and right ($R$) children, XGBoost uses the **gain** — the reduction in objective:

$$\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma$$

- Positive gain → the split improves the objective; grow the tree.
- Non-positive gain → prune; the regularization penalty $\gamma$ outweighs the loss reduction.

**Code:** The gain formula is computed for every candidate split across all features and thresholds.

```python
# xgboost_model.py lines 46–48
gain = 0.5 * (Gl**2 / (Hl + self.reg_lambda) +
              Gr**2 / (Hr + self.reg_lambda) -
              G**2 / (H + self.reg_lambda)) - self.gamma
```

> This is exactly $\frac{1}{2}\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma$. Note `G = Gl + Gr` and `H = Hl + Hr` by construction (line 45).

---

### Split Search

The tree enumerates all midpoints between unique feature values as candidate thresholds.

```python
# xgboost_model.py lines 35–52 (split search loop)
for j in range(n_feats):                        # line 35
    vals = np.unique(X[:, j])                    # line 36
    thresholds = (vals[:-1] + vals[1:]) / 2.0    # line 39 — midpoints
    for t in thresholds:                         # line 40
        mask = X[:, j] <= t                      # line 41 — binary partition
        Gl, Hl = np.sum(g[mask]), np.sum(h[mask])  # line 44
        Gr, Hr = G - Gl, H - Hl                   # line 45
        gain = 0.5 * (...) - self.gamma            # line 46–48
        if gain > best_gain:                       # line 49
            best_gain, best_idx, best_thresh = ...  # lines 50–52
```

> Thresholds are midpoints between sorted unique values. The best split is the one with the maximum gain, subject to `gain > 0` (initial `best_gain = 0.0`, line 32).

---

### Stopping Conditions

A node becomes a leaf (no further splitting) when any of these hold:

```python
# xgboost_model.py lines 29–30 — max depth reached
if depth >= self.max_depth:
    return node

# xgboost_model.py lines 37–38 — feature is constant
if len(vals) < 2:
    continue

# xgboost_model.py lines 42–43 — trivial split (all samples on one side)
if mask.sum() == 0 or mask.sum() == n_samples:
    continue

# xgboost_model.py lines 54–55 — no positive gain found
if best_idx is None:
    return node
```

> These implement the pruning behavior: when $\gamma$ is large or the data is uniform, no split has positive gain and the node stays as a leaf with weight $w^*$.

---

## Prediction (Tree Traversal)

**Code:** Prediction traverses the tree from root to leaf, returning the leaf's optimal weight.

```python
# xgboost_model.py lines 67–72
def _traverse(self, x, node):
    if node.feat_idx is None:                   # line 68 — leaf node
        return node.weight                       # line 69 — return w*
    if x[node.feat_idx] <= node.thresh:          # line 70 — go left
        return self._traverse(x, node.left)      # line 71
    return self._traverse(x, node.right)         # line 72 — go right
```

> At each internal node, the sample is routed left or right based on `x[feat_idx] <= thresh`. The leaf's `node.weight` is $w^* = -G/(H+\lambda)$ as computed during training.

---

## Ensemble Prediction

$$\hat{y} = \hat{y}_{\text{base}} + \eta \sum_{k=1}^{K} f_k(x)$$

```python
# xgboost_model.py lines 99–104
pred = np.full(n, self.base_pred.mean())        # line 101 — base prediction
for tree in self.trees:                          # line 102
    pred += self.lr * tree.predict(X)            # line 103 — accumulate η·f_k(x)
return pred
```

> The final prediction is the base prediction (mean of training labels) plus the learning-rate-scaled sum of all tree predictions.

---

## Summary

| Component | Formula | Code Location |
|---|---|---|
| Objective | $L = \sum l(y_i, \hat{y}_i) + \sum \Omega(f_k)$ | Lines 88–96 (fit loop) |
| Complexity | $\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum w_j^2$ | Lines 17–18 (params) |
| Additive update | $\hat{y}^{(t)} = \hat{y}^{(t-1)} + \eta \, f_t(x)$ | Line 96, 103 |
| Gradient ($g_i$) | $g_i = \hat{y}_i - y_i$ | Line 92 |
| Hessian ($h_i$) | $h_i = 1$ | Line 93 |
| Leaf weight | $w_j^* = -G_j / (H_j + \lambda)$ | Line 27 |
| Split gain | $\frac{1}{2}[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}] - \gamma$ | Lines 46–48 |
| Aggregates | $G_j = \sum g_i, \; H_j = \sum h_i$ | Lines 26, 44–45 |
| Tree traversal | Route left if $x_j \leq t$, else right | Lines 67–72 |
