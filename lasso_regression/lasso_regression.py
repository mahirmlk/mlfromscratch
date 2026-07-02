import numpy as np


class LassoRegression:
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = np.mean(y)

        # precompute column norms squared
        col_sq = np.sum(X ** 2, axis=0)
        col_sq[col_sq == 0] = 1e-12  # avoid div by zero

        for _ in range(self.max_iter):
            w_old = self.w.copy()
            for j in range(d):
                # partial residual excluding feature j
                r_j = y - self.b - X @ self.w + X[:, j] * self.w[j]
                rho_j = X[:, j] @ r_j
                # soft thresholding
                self.w[j] = np.sign(rho_j) * max(abs(rho_j) - self.alpha * n / 2, 0) / col_sq[j]

            if np.max(np.abs(self.w - w_old)) < self.tol:
                break

        return self

    def predict(self, X):
        return X @ self.w + self.b
