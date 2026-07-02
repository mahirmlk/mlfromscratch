import numpy as np
import matplotlib.pyplot as plt
from polynomial_regression import PolynomialRegression

np.random.seed(42)

# generate nonlinear data: y = x^3 - x^2 + noise
n = 200
X = np.random.uniform(-2, 2, (n, 1))
y = X[:, 0]**3 - X[:, 0]**2 + np.random.normal(0, 0.5, n)

# train/test split
idx = np.random.permutation(n)
train, test = idx[:150], idx[150:]
X_train, X_test = X[train], X[test]
y_train, y_test = y[train], y[test]

# fit models
degrees = [2, 3, 4]
models = {}
for d in degrees:
    m = PolynomialRegression(degree=d).fit(X_train, y_train)
    models[d] = m
    y_pred = m.predict(X_test)
    mse = np.mean((y_test - y_pred)**2)
    train_mse = np.mean((y_train - m.predict(X_train))**2)
    print(f"degree={d}  train_mse={train_mse:.4f}  test_mse={mse:.4f}")

# plot
X_plot = np.linspace(-2, 2, 300).reshape(-1, 1)
plt.scatter(X_train[:, 0], y_train, s=10, alpha=0.4, label='train')
plt.scatter(X_test[:, 0], y_test, s=10, alpha=0.4, label='test')
for d, m in models.items():
    plt.plot(X_plot, m.predict(X_plot), label=f'degree={d}')
plt.legend()
plt.title('Polynomial Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.savefig('polynomial_regression/result.png', dpi=120)
plt.show()
