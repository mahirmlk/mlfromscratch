# K-Nearest Neighbors (KNN) — Math & Code Walkthrough

Every formula below is paired with the **exact** line(s) from `knn.py` that implement it.

---

## 1. Model Hyperparameters (`knn.py:5-7`)

```python
def __init__(self, k=5, task='classification'):
    self.k = k                                    # number of neighbors to consider
    self.task = task                              # 'classification' or 'regression'
```

`k` controls the bias–variance trade-off: small `k` → low bias, high variance; large `k` → smoother decision boundaries. The `task` flag selects between majority vote (classification) and averaging (regression).

---

## 2. Lazy Learning — No Training Phase (`knn.py:9-11`)

KNN performs **no training phase** — the entire dataset *is* the model. All computation is deferred to prediction time.

$$\text{model} = \{(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), \dots, (\mathbf{x}_n, y_n)\}$$

```python
def fit(self, X, y):
    self.X_train = np.array(X)                    # line 10: store feature matrix (n × d)
    self.y_train = np.array(y)                    # line 11: store label vector (n,)
```

No parameters are learned. No optimization happens. The "model" is just a pointer to the raw training data.

**Complexity:**
- Training cost: $O(1)$ — just a pointer assignment
- Memory: $O(nd)$ — the entire training set must fit in memory
- Prediction cost: $O(n \cdot d)$ per query (see §3)

---

## 3. Prediction Entry Point (`knn.py:13-15`)

```python
def predict(self, X):
    X = np.array(X)                               # line 14: coerce to numpy array
    return np.array([self._predict_one(x) for x in X])  # line 15: predict each row independently
```

Prediction is embarrassingly parallel — each query point is classified/averaged independently via `_predict_one`.

---

## 4. Distance Metric — Euclidean Distance (`knn.py:18`)

KNN relies on measuring similarity between points. The standard metric is **Euclidean distance**:

$$d(\mathbf{x}, \mathbf{x'}) = \sqrt{\sum_{i=1}^{d}(x_i - x'_i)^2}$$

where $\mathbf{x}$ and $\mathbf{x'}$ are two points in $d$-dimensional space.

```python
dists = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))  # line 18
```

**How it works, step by step:**
1. `self.X_train - x` — NumPy broadcasts `x` (shape `(d,)`) against `self.X_train` (shape `(n, d)`), producing an `(n, d)` matrix of differences
2. `** 2` — element-wise squaring
3. `np.sum(..., axis=1)` — sums across features, yielding `(n,)` vector of squared distances
4. `np.sqrt(...)` — square root gives the final Euclidean distances

The result is a vector of `n` distances, one per training point.

**Complexity:** $O(nd)$ per query — it computes distance to ALL training points. No spatial indexing, no shortcuts. Other common distance metrics include Manhattan ($L_1$) and Minkowski ($L_p$), but this implementation uses only $L_2$.

---

## 5. Finding the k Nearest Neighbors (`knn.py:19-20`)

Given distances to all training points, select the $k$ closest:

$$\mathcal{N}_k(\mathbf{x}_q) = \text{argsort}\big(d(\mathbf{x}_q, \mathbf{x}_1), \dots, d(\mathbf{x}_q, \mathbf{x}_n)\big)[:k]$$

```python
idx = np.argsort(dists)[:self.k]                  # line 19: indices of k smallest distances
neighbors = self.y_train[idx]                     # line 20: retrieve their labels
```

Line 19 sorts all `n$ distances in $O(n \log n)$ and slices the first `k`. Line 20 uses fancy indexing to grab the corresponding labels — an array of `k` class labels (classification) or continuous values (regression).

---

## 6. Classification — Majority Vote (`knn.py:21-23`)

Assign the class $c$ that appears most often among the $k$ nearest neighbors:

$$\hat{y} = \arg\max_{c} \sum_{i=1}^{k} \mathbb{1}[y_{(i)} = c]$$

```python
if self.task == 'classification':                 # line 21: branch on task type
    vals, counts = np.unique(neighbors, return_counts=True)  # line 22: count occurrences of each class
    return vals[np.argmax(counts)]                # line 23: return the most frequent class
```

**How it maps to the formula:**
- `np.unique(neighbors, return_counts=True)` computes $\sum_{i=1}^{k} \mathbb{1}[y_{(i)} = c]$ for every distinct class $c$
- `np.argmax(counts)` finds the $c$ that maximizes this sum — the $\arg\max$
- `vals[argmax]` returns that class label

**Tie-breaking:** When two classes have equal counts, `np.argmax` returns the index of the first occurrence, which corresponds to the smaller class label (since `np.unique` returns sorted values).

---

## 7. Regression — Averaging (`knn.py:24-25`)

For continuous targets, the prediction is the mean of the $k$ nearest neighbors' values:

$$\hat{y} = \frac{1}{k} \sum_{i=1}^{k} y_{(i)}$$

```python
else:                                             # line 24: regression branch
    return np.mean(neighbors)                     # line 25: average of k nearest labels
```

`np.mean(neighbors)` computes exactly $\frac{1}{k}\sum_{i=1}^{k} y_{(i)}$ — the sample mean of the `k` selected labels.

---

## 8. Curse of Dimensionality

As the number of features $d$ grows, KNN degrades because:

1. **Distances concentrate** — the ratio $\frac{d_{\max} - d_{\min}}{d_{\min}} \to 0$, so all points become nearly equidistant.
2. **Volume explodes** — to capture the same fraction of data, the hypercube edge grows as $O(n^{1/d})$, requiring exponentially more samples.
3. **Irrelevant features** dilute the meaningful signal in distance calculations.

**Code connection:** Line 18 computes `sqrt(sum((x - x')^2))` over ALL features equally with no weighting:

```python
dists = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))  # line 18: all features weighted equally
```

If 98 out of 100 features are noise, they dominate the distance and the 2 informative features are drowned out.

**Mitigation:** feature selection, dimensionality reduction (PCA), or using learned distance metrics.

---

## Summary: Line-to-Formula Map

| Lines | Math Concept | Formula |
|-------|-------------|---------|
| 5-7 | Hyperparameters | $k$, task ∈ {classification, regression} |
| 9-11 | Lazy storage | model = raw training data |
| 13-15 | Batch prediction | loop over query points |
| 18 | Euclidean distance | $d(\mathbf{x}, \mathbf{x'}) = \sqrt{\sum(x_i - x'_i)^2}$ |
| 19-20 | k-NN selection | argsort → take top k |
| 21-23 | Majority vote | $\hat{y} = \arg\max_c \sum \mathbb{1}[y_i = c]$ |
| 24-25 | Regression mean | $\hat{y} = \frac{1}{k}\sum y_{(i)}$ |
