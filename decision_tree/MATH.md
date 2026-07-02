# Decision Trees — Mathematical Foundations

## Recursive Binary Splitting

A decision tree partitions the feature space by recursively selecting a feature $j$ and threshold $t$ that best splits the data into two child nodes. At each node, we search for:

$$\arg\min_{j, t} \left[ \frac{n_{left}}{n} \cdot \text{Impurity}(left) + \frac{n_{right}}{n} \cdot \text{Impurity}(right) \right]$$

**Implementation:** The `_build` method iterates over every feature and every unique threshold, computes information gain for each, and keeps the best split:

```python
# decision_tree.py lines 38-45
for feat_idx in range(n_features):
    thresholds = np.unique(X[:, feat_idx])
    for thresh in thresholds:
        gain = self._information_gain(y, X[:, feat_idx], thresh)
        if gain > best_gain:
            best_gain = gain
            best_idx = feat_idx
            best_thresh = thresh
```

After finding the best split, the data is partitioned and the algorithm recurses on each child:

```python
# decision_tree.py lines 50-53
left_mask = X[:, best_idx] <= best_thresh
right_mask = ~left_mask
left = self._build(X[left_mask], y[left_mask], depth + 1)
right = self._build(X[right_mask], y[right_mask], depth + 1)
```

## Gini Impurity (Classification)

Measures the probability of misclassifying a randomly chosen sample:

$$G = 1 - \sum_{k=1}^{K} p_k^2$$

where $p_k$ is the proportion of class $k$ in the node. $G = 0$ means pure (one class only). Maximum is $1 - 1/K$ for uniform class distribution.

**Implementation:** The `_gini` method computes class proportions and applies the formula directly:

```python
# decision_tree.py lines 73-76
def _gini(self, y):
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return 1.0 - np.sum(probs ** 2)
```

`probs` is the vector of $p_k$ values; `np.sum(probs ** 2)` computes $\sum p_k^2$.

## Entropy (Classification)

Information-theoretic measure of disorder:

$$H = -\sum_{k=1}^{K} p_k \log_2 p_k$$

$H = 0$ for a pure node; maximum is $\log_2 K$ for uniform distribution. Using $0 \log 0 = 0$ by convention.

> **Note:** This implementation uses Gini impurity rather than entropy. The impurity dispatcher selects the criterion based on `self.task`:

```python
# decision_tree.py lines 68-71
def _impurity(self, y):
    if self.task == 'classification':
        return self._gini(y)
    return self._mse(y)
```

To use entropy instead, replace `self._gini(y)` with an entropy calculation in this method.

## Information Gain

The reduction in impurity after a split — the core of tree construction:

$$\Delta = \text{Impurity}(parent) - \sum_{j \in \{left, right\}} \frac{n_j}{n} \cdot \text{Impurity}(child_j)$$

Choose the split that maximizes $\Delta$. Equivalent to minimizing weighted child impurity.

**Implementation:** `_information_gain` computes the parent impurity, then the weighted sum of child impurities, and returns the difference:

```python
# decision_tree.py lines 57-66
def _information_gain(self, y, feature, threshold):
    parent = self._impurity(y)
    left_mask = feature <= threshold
    right_mask = ~left_mask
    if left_mask.sum() == 0 or right_mask.sum() == 0:
        return 0
    n = len(y)
    child = (left_mask.sum() / n) * self._impurity(y[left_mask]) + \
            (right_mask.sum() / n) * self._impurity(y[right_mask])
    return parent - child
```

- `parent` = $\text{Impurity}(parent)$
- `left_mask.sum() / n` = $\frac{n_{left}}{n}$
- `self._impurity(y[left_mask])` = $\text{Impurity}(left)$
- The guard on lines 61-62 returns 0 if a split produces an empty child (degenerate split)

## Variance Reduction (Regression)

For regression trees, impurity is measured by variance (or MSE):

$$\text{Var}(S) = \frac{1}{|S|} \sum_{i \in S} (y_i - \bar{y}_S)^2$$

The best split minimizes:

$$\frac{n_{left}}{n} \cdot \text{Var}(left) + \frac{n_{right}}{n} \cdot \text{Var}(right)$$

Leaf predictions are the mean $\bar{y}$ of training samples reaching that leaf.

**Implementation:** The `_mse` method computes $\text{Var}(S) \cdot |S|$ (i.e., sum of squared deviations) as the impurity measure:

```python
# decision_tree.py lines 78-81
def _mse(self, y):
    if len(y) == 0:
        return 0
    return np.var(y) * len(y)  # weighted MSE uses count * var
```

`np.var(y)` computes $\frac{1}{|S|}\sum(y_i - \bar{y})^2$; multiplying by `len(y)` scales it so that the weighted child sum in `_information_gain` produces the correct total squared error reduction.

Regression leaf predictions use the mean:

```python
# decision_tree.py line 87
return np.mean(y)
```

## Stopping Criteria

Growth is halted when any of these hold:

1. **Max depth** reached (configurable via `max_depth`)
2. **Minimum samples** in a node (configurable via `min_samples_split`)
3. Node is **pure** (only one class present)
4. **No positive gain** — split only if $\Delta > 0$

**Implementation:** The stopping conditions are checked at the top of `_build`:

```python
# decision_tree.py lines 32-33
if depth >= self.max_depth or n_classes == 1 or n_samples < self.min_samples_split:
    return Node(value=self._leaf_value(y))
```

And after the search loop, a split is only applied if it improves impurity:

```python
# decision_tree.py lines 47-48
if best_gain <= 0:
    return Node(value=self._leaf_value(y))
```

Without stopping criteria or post-pruning, trees memorize training data (overfit).

## Prediction (Tree Traversal)

At prediction time, a sample is routed from root to leaf by comparing its feature value against each node's threshold:

$$\hat{y} = \text{LeafValue}\Big(\text{descend}(x, \text{root})\Big)$$

where at each internal node: go left if $x_j \leq t$, else go right.

**Implementation:**

```python
# decision_tree.py lines 89-94
def _traverse(self, x, node):
    if node.value is not None:
        return node.value
    if x[node.feature_idx] <= node.threshold:
        return self._traverse(x, node.left)
    return self._traverse(x, node.right)
```

Classification leaves return the majority class; regression leaves return the mean:

```python
# decision_tree.py lines 83-87
def _leaf_value(self, y):
    if self.task == 'classification':
        vals, counts = np.unique(y, return_counts=True)
        return vals[np.argmax(counts)]
    return np.mean(y)
```
