import numpy as np
from collections import Counter


class _Node:
    def __init__(self, feat=None, thresh=None, left=None, right=None, val=None):
        self.feat = feat
        self.thresh = thresh
        self.left = left
        self.right = right
        self.val = val


class _DecisionTree:
    def __init__(self, max_depth=10, max_features=None, task='clf'):
        self.max_depth = max_depth
        self.max_features = max_features
        self.task = task
        self.root = None

    def fit(self, X, y):
        self.root = self._grow(X, y, depth=0)
        return self

    def _grow(self, X, y, depth):
        n_samples, n_feats = X.shape
        if depth >= self.max_depth or n_samples < 2 or len(np.unique(y)) == 1:
            return _Node(val=self._leaf_val(y))

        mf = self.max_features if self.max_features is not None else n_feats
        feat_idx = np.random.choice(n_feats, mf, replace=False)
        if isinstance(feat_idx, np.integer) or not hasattr(feat_idx, '__len__'):
            feat_idx = [feat_idx]
        best_feat, best_thresh, best_gain = None, None, -1
        cur_score = self._score(y)

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

        if best_gain <= 0:
            return _Node(val=self._leaf_val(y))

        mask = X[:, best_feat] <= best_thresh
        left = self._grow(X[mask], y[mask], depth + 1)
        right = self._grow(X[~mask], y[~mask], depth + 1)
        return _Node(feat=best_feat, thresh=best_thresh, left=left, right=right)

    def _score(self, y):
        if self.task == 'clf':
            _, counts = np.unique(y, return_counts=True)
            p = counts / len(y)
            return 1 - np.sum(p ** 2)  # Gini
        else:
            return np.var(y)

    def _leaf_val(self, y):
        if self.task == 'clf':
            return Counter(y).most_common(1)[0][0]
        return np.mean(y)

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node.val is not None:
            return node.val
        if x[node.feat] <= node.thresh:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)


class RandomForest:
    def __init__(self, n_estimators=100, max_depth=10, max_features='sqrt',
                 task='clf', oob_score=False):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.task = task
        self.oob_score = oob_score
        self.trees = []
        self.oob_score_ = None

    def _resolve_max_features(self, n_feats):
        if self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_feats)))
        if self.max_features == 'third':
            return max(1, n_feats // 3)
        return int(self.max_features)

    def fit(self, X, y):
        n, n_feats = X.shape
        mf = self._resolve_max_features(n_feats)
        self.trees = []
        oob_preds = [[] for _ in range(n)]

        for _ in range(self.n_estimators):
            idx = np.random.choice(n, n, replace=True)
            oob_idx = np.setdiff1d(np.arange(n), np.unique(idx))
            tree = _DecisionTree(max_depth=self.max_depth, max_features=mf, task=self.task)
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)
            if self.oob_score and len(oob_idx) > 0:
                preds = tree.predict(X[oob_idx])
                for i, p in zip(oob_idx, preds):
                    oob_preds[i].append(p)

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

        return self

    def predict(self, X):
        preds = np.array([t.predict(X) for t in self.trees])
        if self.task == 'clf':
            out = []
            for col in preds.T:
                out.append(Counter(col).most_common(1)[0][0])
            return np.array(out)
        return np.mean(preds, axis=0)
