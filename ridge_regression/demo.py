import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from ridge_regression import RidgeRegression

np.random.seed(42)

# Generate data with correlated features
n, d = 200, 5
X_base = np.random.randn(n, 2)
X = np.hstack([X_base, X_base[:, :1] + 0.1 * np.random.randn(n, 1),
               X_base[:, 1:2] + 0.1 * np.random.randn(n, 1),
               np.random.randn(n, 1)])
true_w = np.array([3, -2, 1.5, 0.5, -1])
y = X @ true_w + np.random.randn(n) * 0.5

# Split
n_train = int(0.8 * n)
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

# Closed-form: OLS vs Ridge
ols = RidgeRegression(alpha=0.0, solver='closed')
ridge = RidgeRegression(alpha=1.0, solver='closed')
ols.fit(X_train, y_train)
ridge.fit(X_train, y_train)

print("=== Closed-form solver ===")
for name, model in [("OLS", ols), ("Ridge (alpha=1.0)", ridge)]:
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\n{name}: MSE={mse:.4f}, R2={r2:.4f}")
    print(f"  Coefficients: {np.round(model.w, 4)}")

print(f"\nTrue coefficients: {true_w}")

# Gradient descent solver
ridge_gd = RidgeRegression(alpha=1.0, solver='gd', lr=0.001, epochs=5000)
ridge_gd.fit(X_train, y_train)
y_pred = ridge_gd.predict(X_test)
print(f"\n=== Gradient descent solver (alpha=1.0) ===")
print(f"MSE={mean_squared_error(y_test, y_pred):.4f}, R2={r2_score(y_test, y_pred):.4f}")
print(f"Coefficients: {np.round(ridge_gd.w, 4)}")
