"""
Naive Decision Tree: explicit loops, no vectorized impurity computation.

The optimized version (decision_tree.py) uses np.unique, boolean masking,
and vectorized Gini/MSE.  This version does everything with Python dicts
and loops.  Same algorithmic complexity O(n*d*thresholds), but the
constant factor is much larger.

Compare with: decision_tree.py (vectorized NumPy)
Run benchmark: python benchmark.py
"""

import numpy as np


class NodeNaive:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTreeNaive:
    def __init__(self, max_depth=10, min_samples_split=2, task='classification'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.task = task
        self.root = None

    def fit(self, X, y):
        self.root = self._build(X.tolist(), y.tolist(), depth=0)
        return self

    def predict(self, X):
        return np.array([self._traverse(list(x), self.root) for x in X])

    def _build(self, X, y, depth):
        n_samples = len(y)
        n_classes = len(set(y))

        if depth >= self.max_depth or n_classes == 1 or n_samples < self.min_samples_split:
            return NodeNaive(value=self._leaf_value(y))

        best_gain = -1
        best_idx, best_thresh = None, None
        n_features = len(X[0])

        for feat_idx in range(n_features):
            # get unique values for this feature
            seen = set()
            for row in X:
                seen.add(row[feat_idx])
            thresholds = sorted(seen)

            for thresh in thresholds:
                gain = self._information_gain(y, X, feat_idx, thresh)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = feat_idx
                    best_thresh = thresh

        if best_gain <= 0:
            return NodeNaive(value=self._leaf_value(y))

        left_X, left_y, right_X, right_y = [], [], [], []
        for i in range(n_samples):
            if X[i][best_idx] <= best_thresh:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])

        left = self._build(left_X, left_y, depth + 1)
        right = self._build(right_X, right_y, depth + 1)
        return NodeNaive(feature_idx=best_idx, threshold=best_thresh, left=left, right=right)

    def _information_gain(self, y, X, feat_idx, threshold):
        parent = self._impurity(y)
        left_y, right_y = [], []
        for i in range(len(y)):
            if X[i][feat_idx] <= threshold:
                left_y.append(y[i])
            else:
                right_y.append(y[i])
        if len(left_y) == 0 or len(right_y) == 0:
            return 0
        n = len(y)
        child = (len(left_y) / n) * self._impurity(left_y) + \
                (len(right_y) / n) * self._impurity(right_y)
        return parent - child

    def _impurity(self, y):
        if self.task == 'classification':
            return self._gini(y)
        return self._mse(y)

    def _gini(self, y):
        counts = {}
        for label in y:
            counts[label] = counts.get(label, 0) + 1
        n = len(y)
        return 1.0 - sum((c / n) ** 2 for c in counts.values())

    def _mse(self, y):
        if len(y) == 0:
            return 0
        mean = sum(y) / len(y)
        return sum((v - mean) ** 2 for v in y)

    def _leaf_value(self, y):
        if self.task == 'classification':
            counts = {}
            for label in y:
                counts[label] = counts.get(label, 0) + 1
            return max(counts, key=counts.get)
        return sum(y) / len(y)

    def _traverse(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)
