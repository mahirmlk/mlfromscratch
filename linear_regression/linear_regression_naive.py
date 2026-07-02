"""
Naive Gradient Descent: explicit loop version for linear regression.

The optimized version uses vectorized matrix operations (X @ w).
This version computes each gradient component with a Python loop.
Same O(n*d) complexity, but ~50-100x slower constant factor.

Compare with: linear_regression.py (vectorized NumPy)
Run benchmark: python benchmark.py
"""

import numpy as np


class LinearRegressionNaiveGD:
    """Linear regression via gradient descent with explicit Python loops."""

    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        for _ in range(self.epochs):
            # compute predictions with explicit loop
            for i in range(n_samples):
                pred = self.b
                for j in range(n_features):
                    pred += self.w[j] * X[i, j]
                err = pred - y[i]

                # update weights with explicit loop
                for j in range(n_features):
                    self.w[j] -= self.lr * (2.0 / n_samples) * err * X[i, j]
                self.b -= self.lr * (2.0 / n_samples) * err

        return self

    def predict(self, X):
        X = np.array(X)
        return X @ self.w + self.b
