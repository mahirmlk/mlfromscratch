import numpy as np


class KNN:
    def __init__(self, k=5, task='classification'):
        self.k = k
        self.task = task

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x):
        dists = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        idx = np.argsort(dists)[:self.k]
        neighbors = self.y_train[idx]
        if self.task == 'classification':
            vals, counts = np.unique(neighbors, return_counts=True)
            return vals[np.argmax(counts)]
        else:
            return np.mean(neighbors)
