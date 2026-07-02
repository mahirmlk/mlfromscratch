import numpy as np


class TreeNode:
    def __init__(self, depth=0):
        self.left = None
        self.right = None
        self.feat_idx = None
        self.thresh = None
        self.weight = None
        self.depth = depth


class XGBoostTree:
    def __init__(self, max_depth=3, reg_lambda=1.0, gamma=0.0):
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.root = None

    def fit(self, X, g, h):
        self.root = self._build(X, g, h, depth=0)

    def _build(self, X, g, h, depth):
        node = TreeNode(depth=depth)
        G, H = np.sum(g), np.sum(h)
        node.weight = -G / (H + self.reg_lambda)

        if depth >= self.max_depth:
            return node

        best_gain, best_idx, best_thresh = 0.0, None, None
        n_samples, n_feats = X.shape

        for j in range(n_feats):
            vals = np.unique(X[:, j])
            if len(vals) < 2:
                continue
            thresholds = (vals[:-1] + vals[1:]) / 2.0
            for t in thresholds:
                mask = X[:, j] <= t
                if mask.sum() == 0 or mask.sum() == n_samples:
                    continue
                Gl, Hl = np.sum(g[mask]), np.sum(h[mask])
                Gr, Hr = G - Gl, H - Hl
                gain = 0.5 * (Gl**2 / (Hl + self.reg_lambda) +
                              Gr**2 / (Hr + self.reg_lambda) -
                              G**2 / (H + self.reg_lambda)) - self.gamma
                if gain > best_gain:
                    best_gain = gain
                    best_idx = j
                    best_thresh = t

        if best_idx is None:
            return node

        mask = X[:, best_idx] <= best_thresh
        node.feat_idx = best_idx
        node.thresh = best_thresh
        node.left = self._build(X[mask], g[mask], h[mask], depth + 1)
        node.right = self._build(X[~mask], g[~mask], h[~mask], depth + 1)
        return node

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node.feat_idx is None:
            return node.weight
        if x[node.feat_idx] <= node.thresh:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)


class XGBoost:
    def __init__(self, n_estimators=100, learning_rate=0.1,
                 max_depth=3, gamma=0.0, reg_lambda=1.0):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.gamma = gamma
        self.reg_lambda = reg_lambda
        self.trees = []
        self.base_pred = None

    def fit(self, X, y):
        n = len(y)
        self.base_pred = np.full(n, np.mean(y))
        pred = self.base_pred.copy()

        for i in range(self.n_estimators):
            g = pred - y
            h = np.ones(n)
            tree = XGBoostTree(self.max_depth, self.reg_lambda, self.gamma)
            tree.fit(X, g, h)
            pred += self.lr * tree.predict(X)
            self.trees.append(tree)

    def predict(self, X):
        n = X.shape[0]
        pred = np.full(n, self.base_pred.mean())
        for tree in self.trees:
            pred += self.lr * tree.predict(X)
        return pred
