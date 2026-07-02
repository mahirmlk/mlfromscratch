import numpy as np
from elastic_net import ElasticNet

np.random.seed(42)

# generate sparse data
n, p = 200, 20
X = np.random.randn(n, p)
true_w = np.zeros(p)
true_w[:5] = [3, -2, 1.5, -1, 0.5]
y = X @ true_w + np.random.randn(n) * 0.5

# fit elastic net (l1_ratio=0.5 mixes L1 and L2)
en = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)
en.fit(X, y)

# compare with pure ridge (l1_ratio=0)
ridge = ElasticNet(alpha=0.1, l1_ratio=0.0, max_iter=5000)
ridge.fit(X, y)

# compare with pure lasso (l1_ratio=1)
lasso = ElasticNet(alpha=0.1, l1_ratio=1.0, max_iter=5000)
lasso.fit(X, y)

mse_en = np.mean((y - en.predict(X)) ** 2)
mse_ridge = np.mean((y - ridge.predict(X)) ** 2)
mse_lasso = np.mean((y - lasso.predict(X)) ** 2)

print("=== ElasticNet (l1_ratio=0.5) ===")
print(f"MSE: {mse_en:.4f}")
print(f"Non-zero coeffs: {np.sum(np.abs(en.w) > 1e-6)}/{p}")
print(f"Coeffs: {np.round(en.w, 3)}")

print("\n=== Ridge (l1_ratio=0) ===")
print(f"MSE: {mse_ridge:.4f}")
print(f"Non-zero coeffs: {np.sum(np.abs(ridge.w) > 1e-6)}/{p}")
print(f"Coeffs: {np.round(ridge.w, 3)}")

print("\n=== Lasso (l1_ratio=1) ===")
print(f"MSE: {mse_lasso:.4f}")
print(f"Non-zero coeffs: {np.sum(np.abs(lasso.w) > 1e-6)}/{p}")
print(f"Coeffs: {np.round(lasso.w, 3)}")

print("\n=== True Coefficients ===")
print(f"Coeffs: {true_w}")
