import numpy as np


class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # leaf prediction


class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, task='classification'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.task = task
        self.root = None

    def fit(self, X, y):
        self.root = self._build(X, y, depth=0)
        return self

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _build(self, X, y, depth):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # stopping conditions
        if depth >= self.max_depth or n_classes == 1 or n_samples < self.min_samples_split:
            return Node(value=self._leaf_value(y))

        best_gain = -1
        best_idx, best_thresh = None, None

        for feat_idx in range(n_features):
            thresholds = np.unique(X[:, feat_idx])
            for thresh in thresholds:
                gain = self._information_gain(y, X[:, feat_idx], thresh)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = feat_idx
                    best_thresh = thresh

        if best_gain <= 0:
            return Node(value=self._leaf_value(y))

        left_mask = X[:, best_idx] <= best_thresh
        right_mask = ~left_mask
        left = self._build(X[left_mask], y[left_mask], depth + 1)
        right = self._build(X[right_mask], y[right_mask], depth + 1)

        return Node(feature_idx=best_idx, threshold=best_thresh, left=left, right=right)

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

    def _impurity(self, y):
        if self.task == 'classification':
            return self._gini(y)
        return self._mse(y)

    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return 1.0 - np.sum(probs ** 2)

    def _mse(self, y):
        if len(y) == 0:
            return 0
        return np.var(y) * len(y)  # weighted MSE uses count * var

    def _leaf_value(self, y):
        if self.task == 'classification':
            vals, counts = np.unique(y, return_counts=True)
            return vals[np.argmax(counts)]
        return np.mean(y)

    def _traverse(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def tree_depth(self, node=None):
        if node is None:
            node = self.root
        if node.value is not None:
            return 0
        return 1 + max(self.tree_depth(node.left), self.tree_depth(node.right))

    def print_tree(self, node=None, indent=""):
        if node is None:
            node = self.root
        if node.value is not None:
            print(f"{indent}Leaf: {node.value:.4g}")
            return
        print(f"{indent}X[{node.feature_idx}] <= {node.threshold:.4g}")
        print(f"{indent}├─ True:")
        self.print_tree(node.left, indent + "│  ")
        print(f"{indent}└─ False:")
        self.print_tree(node.right, indent + "   ")
