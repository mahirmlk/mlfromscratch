import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from xgboost_model import XGBoost


def make_data(n=500, d=5, noise=0.3, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    w = rng.randn(d)
    y = X @ w + noise * rng.randn(n)
    return X, y


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


if __name__ == '__main__':
    X_train, y_train = make_data(n=200)
    X_test, y_test = make_data(n=50, seed=99)

    xgb = XGBoost(n_estimators=50, learning_rate=0.1,
                  max_depth=3, gamma=0.1, reg_lambda=1.0)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    print(f"XGBoost MSE: {mse(y_test, y_pred):.4f}")

    # compare with fewer trees
    xgb2 = XGBoost(n_estimators=10, learning_rate=0.3,
                   max_depth=2, gamma=0.0, reg_lambda=0.0)
    xgb2.fit(X_train, y_train)
    y_pred2 = xgb2.predict(X_test)
    print(f"XGBoost (10 trees, deeper lr) MSE: {mse(y_test, y_pred2):.4f}")
