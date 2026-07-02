import numpy as np


class ElasticNet:
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        n, p = X.shape
        self.w = np.zeros(p)
        self.b = np.mean(y)
        r = y - self.b
        # precompute column norms
        col_sq = np.sum(X ** 2, axis=0) / n
        l1 = self.alpha * self.l1_ratio
        l2 = self.alpha * (1 - self.l1_ratio)

        for _ in range(self.max_iter):
            w_old = self.w.copy()
            for j in range(p):
                # partial residual without feature j
                r += X[:, j] * self.w[j]
                # coordinate update with soft-thresholding + L2 shrinkage
                rho = np.dot(X[:, j], r) / n
                denom = col_sq[j] + l2
                self.w[j] = self._soft(rho, l1) / denom
                r -= X[:, j] * self.w[j]

            # intercept
            self.b = np.mean(r)

            if np.max(np.abs(self.w - w_old)) < self.tol:
                break
        return self

    def predict(self, X):
        return X @ self.w + self.b

    def _soft(self, x, t):
        return np.sign(x) * np.maximum(np.abs(x) - t, 0)
