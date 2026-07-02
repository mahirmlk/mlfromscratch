# Random Forest — Mathematical Foundations

## 1. Bootstrap Aggregating (Bagging)

Given a training set $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}$, we create $B$ bootstrap samples $\mathcal{D}_1, \mathcal{D}_2, \dots, \mathcal{D}_B$ by sampling $N$ points **with replacement** from $\mathcal{D}$.

Each bootstrap sample contains on average $\approx 63.2\%$ of unique original points:

$$P(\text{point } i \text{ not sampled}) = \left(1 - \frac{1}{N}\right)^N \approx e^{-1} \approx 0.368$$

```python
# random_forest.py lines 106-109
idx = np.random.choice(n, n, replace=True)
oob_idx = np.setdiff1d(np.arange(n), np.unique(idx))
tree = _DecisionTree(max_depth=self.max_depth, max_features=mf, task=self.task)
tree.fit(X[idx], y[idx])
```
`np.random.choice(n, n, replace=True)` draws $N$ samples with replacement to form $\mathcal{D}_b$; `np.setdiff1d` computes the OOB indices (points not in $\mathcal{D}_b$).

A decision tree $T_b$ is trained on each bootstrap sample $\mathcal{D}_b$.

## 2. Random Feature Subset at Each Split

At each internal node, instead of evaluating all $p$ features, we select a random subset of $m$ features ($m < p$) and find the best split only among those.

- **Classification:** $m = \sqrt{p}$ (typical default)
- **Regression:** $m = p/3$

```python
# random_forest.py lines 92-97
def _resolve_max_features(self, n_feats):
    if self.max_features == 'sqrt':
        return max(1, int(np.sqrt(n_feats)))
    if self.max_features == 'third':
        return max(1, n_feats // 3)
    return int(self.max_features)
```
`sqrt` → $m = \lfloor\sqrt{p}\rfloor$, `third` → $m = \lfloor p/3 \rfloor$.

```python
# random_forest.py lines 30-31
mf = self.max_features if self.max_features is not None else n_feats
feat_idx = np.random.choice(n_feats, mf, replace=False)
```
`np.random.choice(n_feats, mf, replace=False)` draws $m$ distinct feature indices from $\{0, \dots, p-1\}$.

This decorrelates the trees. If one strong predictor dominates, standard bagged trees would all split on it first — random feature selection forces diversity.

## 3. Information Gain (Split Criterion)

For each candidate split $(j, t)$, we compute the **information gain**:

$$\Delta I = I(\mathcal{D}) - \frac{|\mathcal{D}_L|}{|\mathcal{D}|}\,I(\mathcal{D}_L) - \frac{|\mathcal{D}_R|}{|\mathcal{D}|}\,I(\mathcal{D}_R)$$

where $I$ is the node impurity (Gini for classification, variance for regression).

### Gini Impurity (Classification)

$$G = 1 - \sum_{c=1}^{C} p_c^2$$

where $p_c$ is the proportion of class $c$ in the node.

```python
# random_forest.py lines 57-61
def _score(self, y):
    if self.task == 'clf':
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return 1 - np.sum(p ** 2)  # Gini
```
`1 - np.sum(p ** 2)` computes $G = 1 - \sum p_c^2$ directly from class proportions.

### Variance (Regression)

$$\text{Var}(y) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \bar{y})^2$$

```python
# random_forest.py lines 62-63
    else:
        return np.var(y)
```
`np.var(y)` computes the population variance of the node's targets.

### Gain Computation Over All Candidate Splits

```python
# random_forest.py lines 37-47
for f in feat_idx:
    thresholds = np.unique(X[:, f])
    for t in thresholds:
        left = y[X[:, f] <= t]
        right = y[X[:, f] > t]
        if len(left) == 0 or len(right) == 0:
            continue
        gain = cur_score - (len(left) / n_samples * self._score(left)
                            + len(right) / n_samples * self._score(right))
        if gain > best_gain:
            best_feat, best_thresh, best_gain = f, t, gain
```
This loops over all selected features $f \in \{m \text{ random features}\}$ and unique thresholds $t$, computing $\Delta I$ for each. The split with the highest gain is chosen.

## 4. Prediction

**Classification** — majority vote (mode):

$$\hat{y} = \arg\max_c \sum_{b=1}^{B} \mathbb{I}\!\left[T_b(x) = c\right]$$

Equivalently: $\hat{y} = \text{mode}\{T_1(x), T_2(x), \dots, T_B(x)\}$

```python
# random_forest.py lines 132-138
def predict(self, X):
    preds = np.array([t.predict(X) for t in self.trees])
    if self.task == 'clf':
        out = []
        for col in preds.T:
            out.append(Counter(col).most_common(1)[0][0])
        return np.array(out)
```
`Counter(col).most_common(1)[0][0]` returns the mode across all $B$ tree predictions for each sample — the majority vote.

**Regression** — average:

$$\hat{y} = \frac{1}{B} \sum_{b=1}^{B} T_b(x)$$

```python
# random_forest.py line 139
    return np.mean(preds, axis=0)
```
`np.mean(preds, axis=0)` averages the $B$ tree predictions element-wise.

## 5. Leaf Value (Terminal Node Output)

**Classification** — most frequent class:

$$\hat{y}_{\text{leaf}} = \text{mode}(y_{\text{leaf}})$$

```python
# random_forest.py lines 65-67
def _leaf_val(self, y):
    if self.task == 'clf':
        return Counter(y).most_common(1)[0][0]
```

**Regression** — mean of targets in the leaf:

$$\hat{y}_{\text{leaf}} = \frac{1}{|\text{leaf}|}\sum_{i \in \text{leaf}} y_i$$

```python
# random_forest.py line 68
    return np.mean(y)
```

## 6. Tree Traversal (Prediction Path)

Given a new sample $x$, traverse the tree: at each internal node, go left if $x_j \leq t$, else go right, until a leaf is reached.

```python
# random_forest.py lines 73-78
def _traverse(self, x, node):
    if node.val is not None:
        return node.val
    if x[node.feat] <= node.thresh:
        return self._traverse(x, node.left)
    return self._traverse(x, node.right)
```
The split rule $x_j \leq t$ at line 76 routes the sample left or right recursively until a leaf (`node.val is not None`) is reached.

## 7. Stopping Criteria (Tree Growth)

A node becomes a leaf (stops splitting) when any of these hold:

1. Maximum depth reached: $\text{depth} \geq d_{\max}$
2. Too few samples: $n_{\text{node}} < 2$
3. Pure node: all labels identical ($|\text{unique}(y)| = 1$)
4. No positive gain: $\Delta I^* \leq 0$

```python
# random_forest.py lines 27-28 (criteria 1-3)
if depth >= self.max_depth or n_samples < 2 or len(np.unique(y)) == 1:
    return _Node(val=self._leaf_val(y))

# random_forest.py lines 49-50 (criterion 4)
if best_gain <= 0:
    return _Node(val=self._leaf_val(y))
```

## 8. Out-of-Bag (OOB) Error Estimation

Each tree $T_b$ is built on $\mathcal{D}_b$. For any point $x_i$, roughly $36.8\%$ of trees **did not** see it during training. These are the OOB trees for $x_i$:

$$\text{OOB}(x_i) = \frac{1}{|\{b : x_i \notin \mathcal{D}_b\}|} \sum_{b:\, x_i \notin \mathcal{D}_b} T_b(x_i)$$

```python
# random_forest.py lines 107, 111-114
oob_idx = np.setdiff1d(np.arange(n), np.unique(idx))
...
if self.oob_score and len(oob_idx) > 0:
    preds = tree.predict(X[oob_idx])
    for i, p in zip(oob_idx, preds):
        oob_preds[i].append(p)
```
`oob_idx` identifies points not in $\mathcal{D}_b$; each tree's predictions for its OOB points are accumulated in `oob_preds[i]`.

The OOB error aggregates predictions only from trees that never trained on each point — providing a built-in cross-validation estimate **without** a separate validation set.

$$\text{OOB Error} = \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}\!\left(y_i,\; \text{OOB}(x_i)\right)$$

```python
# random_forest.py lines 116-128
if self.oob_score:
    correct = 0
    count = 0
    for i in range(n):
        if len(oob_preds[i]) > 0:
            count += 1
            if self.task == 'clf':
                pred = Counter(oob_preds[i]).most_common(1)[0][0]
            else:
                pred = np.mean(oob_preds[i])
            if pred == y[i]:
                correct += 1
    self.oob_score_ = correct / count if count > 0 else None
```
For classification, OOB prediction is the mode of accumulated OOB tree votes; for regression, the mean. OOB score = `correct / count` (accuracy-based $\mathcal{L}$).

## 9. Why Random Forest Reduces Variance

For $B$ identically distributed predictors with variance $\sigma^2$ and pairwise correlation $\rho$:

$$\text{Var}\!\left(\frac{1}{B}\sum_{b=1}^{B} T_b\right) = \rho\,\sigma^2 + \frac{1-\rho}{B}\,\sigma^2$$

- **Bagging** ($\rho = 1$): variance = $\sigma^2$ — no reduction.
- **Random Forest** (adds feature randomness → reduces $\rho$): the $\rho\,\sigma^2$ term shrinks, and as $B \to \infty$ the second term vanishes.

**Key insight:** Reducing correlation $\rho$ between trees is more effective than simply increasing $B$. Random feature selection at each split is precisely what drives $\rho$ down, making Random Forests far more powerful than plain bagged trees.
