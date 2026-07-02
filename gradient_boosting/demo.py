import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from gradient_boosting import GradientBoosting

np.random.seed(42)

n = 200
X = np.random.randn(n, 5)
y = 3 * np.sin(X[:, 0]) + X[:, 1] ** 2 - X[:, 2] + 0.5 * np.random.randn(n)

split = 160
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = GradientBoosting(n_estimators=100, learning_rate=0.1, max_depth=3)
model.fit(X_train, y_train)

preds = model.predict(X_test)
mse = np.mean((y_test - preds) ** 2)
print(f"Test MSE: {mse:.4f}")
print(f"Final train MSE: {model.train_losses[-1]:.4f}")
print(f"Loss improved: {model.train_losses[0]:.4f} -> {model.train_losses[-1]:.4f}")
