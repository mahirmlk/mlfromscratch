import numpy as np


class LogisticRegression:
    def __init__(self, lr=0.01, epochs=1000, multi_class='ovr'):
        self.lr = lr
        self.epochs = epochs
        self.multi_class = multi_class

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def _log_loss(self, y, p):
        eps = 1e-15
        return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        if n_classes == 2:
            self._fit_binary(X, y)
        else:
            self._fit_multi(X, y, n_classes)
        return self

    def _fit_binary(self, X, y):
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0
        for _ in range(self.epochs):
            z = X @ self.w + self.b
            p = self._sigmoid(z)
            # gradient of log loss
            dw = (1 / m) * (X.T @ (p - y))
            db = (1 / m) * np.sum(p - y)
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def _fit_multi(self, X, y, n_classes):
        self.models_ = []
        for c in self.classes_:
            y_bin = (y == c).astype(float)
            m = LogisticRegression(lr=self.lr, epochs=self.epochs)
            m._fit_binary(X, y_bin)
            self.models_.append(m)

    def predict_proba(self, X):
        X = np.array(X)
        if hasattr(self, 'models_'):
            probs = np.column_stack([m._sigmoid(X @ m.w + m.b) for m in self.models_])
            return probs / probs.sum(axis=1, keepdims=True)
        return self._sigmoid(X @ self.w + self.b)

    def predict(self, X):
        if hasattr(self, 'models_'):
            probs = self.predict_proba(X)
            return self.classes_[np.argmax(probs, axis=1)]
        return (self.predict_proba(X) >= 0.5).astype(int)
