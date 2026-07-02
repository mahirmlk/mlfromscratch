import numpy as np
from itertools import combinations_with_replacement


class PolynomialRegression:
    def __init__(self, degree=2):
        self.degree = degree
        self.coef = None

    def _expand(self, X):
        n, d = X.shape
        features = [np.ones(n)]
        for deg in range(1, self.degree + 1):
            for combo in combinations_with_replacement(range(d), deg):
                feat = np.prod(X[:, combo], axis=1)
                features.append(feat)
        return np.column_stack(features)

    def fit(self, X, y):
        Phi = self._expand(X)
        # normal equation: w = (Phi^T Phi)^-1 Phi^T y
        self.coef = np.linalg.lstsq(Phi, y, rcond=None)[0]
        return self

    def predict(self, X):
        return self._expand(X) @ self.coef
