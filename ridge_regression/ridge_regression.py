import numpy as np


class RidgeRegression:
    def __init__(self, alpha=1.0, solver='closed', lr=0.01, epochs=1000):
        self.alpha = alpha
        self.solver = solver
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        n, d = X.shape
        if self.solver == 'closed':
            # w = (X^T X + alpha I)^{-1} X^T y
            I = np.eye(d)
            self.w = np.linalg.solve(X.T @ X + self.alpha * I, X.T @ y)
            self.b = np.mean(y - X @ self.w)
        else:
            self.w = np.zeros(d)
            self.b = 0.0
            for _ in range(self.epochs):
                y_pred = X @ self.w + self.b
                err = y_pred - y
                # grad_w = (2/n) X^T err + (2*alpha/n) w
                self.w -= self.lr * ((2.0 / n) * (X.T @ err) + (2.0 * self.alpha / n) * self.w)
                self.b -= self.lr * (2.0 / n) * np.sum(err)
        return self

    def predict(self, X):
        return X @ self.w + self.b
