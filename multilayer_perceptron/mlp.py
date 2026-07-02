import numpy as np


class MLP:
    def __init__(self, hidden_layers=(64, 32), lr=0.01, epochs=100, batch_size=32):
        self.hidden_layers = hidden_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.W, self.b = [], []

    def _init_weights(self, layer_sizes):
        self.W = [np.random.randn(n_in, n_out) * 0.01
                  for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.b = [np.zeros((1, n_out)) for n_out in layer_sizes[1:]]

    @staticmethod
    def _relu(z):
        return np.maximum(0, z)

    @staticmethod
    def _relu_deriv(a):
        return (a > 0).astype(float)

    @staticmethod
    def _softmax(z):
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def _forward(self, X):
        a = X
        acts, zs = [a], []
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            zs.append(z)
            if i < len(self.W) - 1:
                a = self._relu(z)
            else:
                a = self._softmax(z)  # always softmax for output
            acts.append(a)
        return acts, zs

    def _one_hot(self, y):
        oh = np.zeros((len(y), self.n_classes))
        oh[np.arange(len(y)), y] = 1
        return oh

    def _backward(self, acts, zs, y):
        m = len(y)
        n_layers = len(self.W)
        dW, db = [None] * n_layers, [None] * n_layers

        # softmax + cross-entropy gradient
        dz = acts[-1] - self._one_hot(y)

        for i in reversed(range(n_layers)):
            dW[i] = acts[i].T @ dz / m
            db[i] = dz.sum(axis=0, keepdims=True) / m
            if i > 0:
                dz = (dz @ self.W[i].T) * self._relu_deriv(acts[i])

        for i in range(n_layers):
            self.W[i] -= self.lr * dW[i]
            self.b[i] -= self.lr * db[i]

    def _loss(self, pred, y):
        eps = 1e-12
        # cross-entropy with one-hot
        return -np.mean(np.log(pred[np.arange(len(y)), y] + eps))

    def fit(self, X, y):
        y = y.ravel().astype(int)
        self.n_classes = len(np.unique(y))
        if self.n_classes == 2:
            self.n_classes = 2  # keep 2 output neurons for softmax
        sizes = [X.shape[1]] + list(self.hidden_layers) + [self.n_classes]
        self._init_weights(sizes)
        self.losses = []

        for _ in range(self.epochs):
            idx = np.random.permutation(len(X))
            for start in range(0, len(X), self.batch_size):
                bi = idx[start:start + self.batch_size]
                acts, zs = self._forward(X[bi])
                self._backward(acts, zs, y[bi])
            acts, _ = self._forward(X)
            self.losses.append(self._loss(acts[-1], y))
        return self

    def predict(self, X):
        acts, _ = self._forward(X)
        return acts[-1].argmax(axis=1)

    def predict_proba(self, X):
        acts, _ = self._forward(X)
        return acts[-1]
