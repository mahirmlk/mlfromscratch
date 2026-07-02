import numpy as np


class GaussianNB:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.theta_ = np.zeros((n_classes, n_features))  # means
        self.sigma_ = np.zeros((n_classes, n_features))   # variances
        self.class_prior_ = np.zeros(n_classes)

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.theta_[i] = X_c.mean(axis=0)
            self.sigma_[i] = X_c.var(axis=0) + self.var_smoothing
            self.class_prior_[i] = X_c.shape[0] / X.shape[0]

        return self

    def _log_gaussian(self, X, mean, var):
        # log N(x | mu, sigma) per feature
        return -0.5 * (np.log(2.0 * np.pi * var) + ((X - mean) ** 2) / var)

    def predict(self, X):
        log_post = np.zeros((X.shape[0], len(self.classes_)))
        for i, c in enumerate(self.classes_):
            lp = self._log_gaussian(X, self.theta_[i], self.sigma_[i])
            log_post[:, i] = np.log(self.class_prior_[i]) + lp.sum(axis=1)
        return self.classes_[np.argmax(log_post, axis=1)]

    def predict_proba(self, X):
        log_post = np.zeros((X.shape[0], len(self.classes_)))
        for i, c in enumerate(self.classes_):
            lp = self._log_gaussian(X, self.theta_[i], self.sigma_[i])
            log_post[:, i] = np.log(self.class_prior_[i]) + lp.sum(axis=1)
        # softmax in log-space
        max_lp = log_post.max(axis=1, keepdims=True)
        exp_post = np.exp(log_post - max_lp)
        return exp_post / exp_post.sum(axis=1, keepdims=True)
