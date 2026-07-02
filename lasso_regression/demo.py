import numpy as np
from lasso_regression import LassoRegression

np.random.seed(42)

# sparse true coefficients: only 4 out of 20 features are relevant
n, d = 200, 20
X = np.random.randn(n, d)
true_w = np.zeros(d)
true_w[0] = 3.0
true_w[5] = -2.0
true_w[10] = 1.5
true_w[15] = -4.0
y = X @ true_w + np.random.randn(n) * 0.5

# Lasso
lasso = LassoRegression(alpha=0.5, max_iter=2000)
lasso.fit(X, y)
y_pred_lasso = lasso.predict(X)
mse_lasso = np.mean((y - y_pred_lasso) ** 2)
sparsity = np.sum(np.abs(lasso.w) < 1e-6)

# OLS (unregularized via Lasso with alpha=0)
ols = LassoRegression(alpha=0.0, max_iter=2000)
ols.fit(X, y)
y_pred_ols = ols.predict(X)
mse_ols = np.mean((y - y_pred_ols) ** 2)

print("True nonzero indices:  [0, 5, 10, 15]")
print(f"Lasso nonzero indices: {np.where(np.abs(lasso.w) > 1e-6)[0].tolist()}")
print(f"Lasso coefficients:    {np.round(lasso.w, 3)}")
print(f"\nLasso MSE: {mse_lasso:.4f}")
print(f"OLS   MSE: {mse_ols:.4f}")
print(f"Zero coefficients (Lasso): {sparsity}/{d}")
print(f"Zero coefficients (OLS):   {np.sum(np.abs(ols.w) < 1e-6)}/{d}")
