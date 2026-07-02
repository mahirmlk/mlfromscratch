import numpy as np


class DecisionStump:
    def __init__(self):
        self.feat_idx = None
        self.thresh = None
        self.left_val = 1
        self.right_val = -1

    def predict(self, X):
        n = X.shape[0]
        preds = np.ones(n)
        preds[X[:, self.feat_idx] < self.thresh] = self.left_val
        preds[X[:, self.feat_idx] >= self.thresh] = self.right_val
        return preds


class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.stumps = []
        self.alphas = []

    def fit(self, X, y):
        n = X.shape[0]
        w = np.ones(n) / n
        self.stumps = []
        self.alphas = []

        for _ in range(self.n_estimators):
            stump = self._best_stump(X, y, w)
            preds = stump.predict(X)
            miss = preds != y
            err = np.sum(w * miss) / np.sum(w)
            err = np.clip(err, 1e-10, 1 - 1e-10)

            alpha = 0.5 * np.log((1 - err) / err)
            w *= np.exp(-alpha * y * preds)
            w /= np.sum(w)

            self.stumps.append(stump)
            self.alphas.append(alpha)

    def predict(self, X):
        agg = sum(a * s.predict(X) for a, s in zip(self.alphas, self.stumps))
        return np.sign(agg).astype(int)

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
