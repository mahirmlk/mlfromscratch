import numpy as np


class BayesianRegression:
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta
        self.m = None
        self.S = None
        self.X_mean = None
        self.X_std = None

    def _preprocess(self, X, fit=False):
        if fit:
            self.X_mean = X.mean(axis=0)
            self.X_std = X.std(axis=0) + 1e-8
        return (X - self.X_mean) / self.X_std

    def _add_bias(self, X):
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def fit(self, X, y):
        Xp = self._preprocess(X, fit=True)
        Xb = self._add_bias(Xp)
        d = Xb.shape[1]
        S_inv = self.alpha * np.eye(d) + self.beta * Xb.T @ Xb
        self.S = np.linalg.inv(S_inv)
        self.m = self.beta * self.S @ Xb.T @ y
        return self

    def predict(self, X):
        Xp = self._preprocess(X)
        Xb = self._add_bias(Xp)
        return Xb @ self.m

    def predict_with_uncertainty(self, X):
        Xp = self._preprocess(X)
        Xb = self._add_bias(Xp)
        mu = Xb @ self.m
        var = 1.0 / self.beta + np.sum(Xb @ self.S * Xb, axis=1)
        return mu, var
