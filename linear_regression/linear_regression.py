import numpy as np


class LinearRegression:
    def __init__(self, solver='closed_form', lr=0.01, epochs=1000):
        self.solver = solver
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1)
        m, n = X.shape

        if self.solver == 'closed_form':
            # normal equation: w = (X^T X)^-1 X^T y
            X_b = np.hstack([np.ones((m, 1)), X])
            theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
            self.b = theta[0]
            self.w = theta[1:]
        else:
            self.w = np.zeros(n)
            self.b = 0.0
            for _ in range(self.epochs):
                y_pred = X @ self.w + self.b
                err = y_pred - y
                # gradient step
                grad_w = (2 / m) * (X.T @ err)
                grad_b = (2 / m) * np.sum(err)
                self.w -= self.lr * grad_w
                self.b -= self.lr * grad_b

    def predict(self, X):
        X = np.array(X)
        return X @ self.w + self.b
