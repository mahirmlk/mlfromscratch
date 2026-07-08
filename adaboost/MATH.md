# AdaBoost

## Overview

AdaBoost (Adaptive Boosting) combines multiple weak classifiers into a strong classifier by iteratively reweighting training samples so that misclassified points receive more attention.

## Algorithm

**Input:** Training set $\{(x_1, y_1), \dots, (x_N, y_N)\}$ where $y_i \in \{-1, +1\}$

### 1. Initialize Sample Weights

$$w_i^{(1)} = \frac{1}{N}, \quad i = 1, \dots, N$$

```python
# adaboost.py, line 27
w = np.ones(n) / n
```

Each sample starts with equal weight $1/N$, so no point is initially more important than another.

---

### 2. For each round $t = 1, \dots, T$

```python
# adaboost.py, line 31
for _ in range(self.n_estimators):
```

The outer loop runs for `n_estimators` rounds (default 50).

---

**a) Train weak learner** $h_t$ on the weighted dataset.

```python
# adaboost.py, line 32
stump = self._best_stump(X, y, w)
```

The weak learner is a `DecisionStump` — a single-split tree that picks the best feature, threshold, and direction. The full search is in `_best_stump`:

```python
# adaboost.py, lines 49-69
def _best_stump(self, X, y, w):
    n, d = X.shape
    best = DecisionStump()
    best_err = np.inf
    for j in range(d):
        vals = np.unique(X[:, j])
        threshs = (vals[:-1] + vals[1:]) / 2 if len(vals) > 1 else vals
        for t in threshs:
            for lv, rv in [(1, -1), (-1, 1)]:
                preds = np.ones(n)
                preds[X[:, j] < t] = lv
                preds[X[:, j] >= t] = rv
                err = np.sum(w * (preds != y))
                if err < best_err:
                    best_err = err
                    best.feat_idx = j
                    best.thresh = t
                    best.left_val = lv
                    best.right_val = rv
    return best
```

This exhaustively searches all features, all mid-point thresholds, and both polarity directions, selecting the stump that minimizes the **weighted** error $\sum_i w_i \cdot \mathbf{1}[h(x_i) \neq y_i]$.

The stump's prediction rule:

```python
# adaboost.py, lines 14-15
preds[X[:, self.feat_idx] < self.thresh] = self.left_val
preds[X[:, self.feat_idx] >= self.thresh] = self.right_val
```

---

**b) Compute weighted error:**

$$\epsilon_t = \frac{\sum_{i:\, h_t(x_i) \neq y_i} w_i^{(t)}}{\sum_{i=1}^{N} w_i^{(t)}}$$

```python
# adaboost.py, lines 33-36
preds = stump.predict(X)
miss = preds != y
err = np.sum(w * miss) / np.sum(w)
err = np.clip(err, 1e-10, 1 - 1e-10)
```

The indicator $\mathbf{1}[h_t(x_i) \neq y_i]$ is computed as `miss = preds != y`. The dot product `w * miss` sums only the weights of misclassified samples. The `np.clip` prevents division-by-zero in the log when computing $\alpha_t$.

---

**c) Compute classifier weight** (higher for more accurate classifiers):

$$\alpha_t = \frac{1}{2} \ln \frac{1 - \epsilon_t}{\epsilon_t}$$

```python
# adaboost.py, line 38
alpha = 0.5 * np.log((1 - err) / err)
```

Classifiers with lower error get larger $\alpha_t$, meaning more influence in the final vote.

---

**d) Update sample weights** (increase weight of misclassified samples):

$$w_i^{(t+1)} = w_i^{(t)} \exp\!\bigl(-\alpha_t \, y_i \, h_t(x_i)\bigr)$$

Then normalize so $\sum_i w_i^{(t+1)} = 1$.

```python
# adaboost.py, lines 39-40
w *= np.exp(-alpha * y * preds)
w /= np.sum(w)
```

When $h_t(x_i) = y_i$, the exponent $-\alpha_t y_i h_t(x_i) = -\alpha_t < 0$, so the weight **decreases**. When $h_t(x_i) \neq y_i$, the exponent is $+\alpha_t > 0$, so the weight **increases**. This forces the next learner to focus on previously misclassified points.

---

**Store the weak learner and its weight:**

```python
# adaboost.py, lines 42-43
self.stumps.append(stump)
self.alphas.append(alpha)
```

---

### 3. Final Prediction

$$H(x) = \operatorname{sign}\!\left(\sum_{t=1}^{T} \alpha_t \, h_t(x)\right)$$

```python
# adaboost.py, lines 46-47
agg = sum(a * s.predict(X) for a, s in zip(self.alphas, self.stumps))
return np.sign(agg).astype(int)
```

Each stump's prediction is scaled by its classifier weight $\alpha_t$, then summed. The `sign` function maps the weighted vote to $\{-1, +1\}$.

---

## Intuition

| Component | Role |
|---|---|
| $\epsilon_t$ | Measures how wrong the weak classifier is on the *weighted* data |
| $\alpha_t$ | Gives more voting power to classifiers with lower error |
| $w_i^{(t+1)}$ | Forces the next learner to focus on previously misclassified points |

- When $\epsilon_t < 0.5$ (better than random), $\alpha_t > 0$ — the classifier contributes positively.
- When $\epsilon_t = 0.5$ (random guessing), $\alpha_t = 0$ — the classifier is ignored.
- When $\epsilon_t > 0.5$ (worse than random), $\alpha_t < 0$ — the classifier's vote is inverted.

## Training Error Bound

The training error of the combined classifier is bounded by:

$$\frac{1}{N} \sum_{i=1}^{N} \mathbf{1}[H(x_i) \neq y_i] \;\leq\; \prod_{t=1}^{T} 2\sqrt{\epsilon_t(1 - \epsilon_t)}$$

This exponential decay in the bound is what drives AdaBoost's strong generalization when weak learners are consistently better than chance.
