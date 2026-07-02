import numpy as np


class TreeNode:
    def __init__(self):
        self.left = None
        self.right = None
        self.feat_idx = None
        self.thresh = None
        self.val = None


class RegressionTree:
    def __init__(self, max_depth=3, min_samples_leaf=5):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    def fit(self, X, y):
        self.root = self._build(X, y, depth=0)
        return self

    def _build(self, X, y, depth):
        node = TreeNode()
        n = len(y)
        if depth >= self.max_depth or n <= 2 * self.min_samples_leaf:
            node.val = np.mean(y)
            return node

        best_gain = -np.inf
        best_feat, best_thresh = None, None
        current_var = np.var(y) * n

        for j in range(X.shape[1]):
            vals = np.unique(X[:, j])
            if len(vals) < 2:
                continue
            thresholds = (vals[:-1] + vals[1:]) / 2
            for t in thresholds:
                mask = X[:, j] <= t
                nl, nr = mask.sum(), n - mask.sum()
                if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                    continue
                gain = current_var - np.var(y[mask]) * nl - np.var(y[~mask]) * nr
                if gain > best_gain:
                    best_gain = gain
                    best_feat = j
                    best_thresh = t

        if best_feat is None or best_gain <= 0:
            node.val = np.mean(y)
            return node

        node.feat_idx = best_feat
        node.thresh = best_thresh
        mask = X[:, best_feat] <= best_thresh
        node.left = self._build(X[mask], y[mask], depth + 1)
        node.right = self._build(X[~mask], y[~mask], depth + 1)
        return node

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node.val is not None:
            return node.val
        if x[node.feat_idx] <= node.thresh:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)


class GradientBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.train_losses = []

    def fit(self, X, y):
        n = len(y)
        pred = np.zeros(n)
        self.init_val = np.mean(y)
        pred[:] = self.init_val
        self.trees = []
        self.train_losses = []

        for _ in range(self.n_estimators):
            resid = y - pred
            tree = RegressionTree(max_depth=self.max_depth)
            tree.fit(X, resid)
            pred += self.lr * tree.predict(X)
            self.trees.append(tree)
            self.train_losses.append(np.mean((y - pred) ** 2))

    def predict(self, X):
        pred = np.full(X.shape[0], self.init_val)
        for tree in self.trees:
            pred += self.lr * tree.predict(X)
        return pred
