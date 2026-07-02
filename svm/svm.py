import numpy as np


class SVM:
    def __init__(self, C=1.0, lr=0.001, epochs=1000):
        self.C = C
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0

        for _ in range(self.epochs):
            for i in range(n):
                margin = y[i] * (np.dot(X[i], self.w) + self.b)
                if margin < 1:
                    self.w -= self.lr * (self.w / self.C - y[i] * X[i])
                    self.b -= self.lr * (-y[i])
                else:
                    self.w -= self.lr * self.w / self.C

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)
