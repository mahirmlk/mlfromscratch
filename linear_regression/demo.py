import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from linear_regression import LinearRegression

np.random.seed(42)

# synthetic data: y = 3*x + 7 + noise
X = np.random.randn(200, 1) * 10
y = 3 * X.squeeze() + 7 + np.random.randn(200) * 2

# manual train/test split
idx = np.random.permutation(200)
train, test = idx[:160], idx[160:]
X_train, X_test = X[train], X[test]
y_train, y_test = y[train], y[test]

# closed form
model = LinearRegression(solver='closed_form')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("=== Closed Form ===")
print(f"w = {model.w}, b = {model.b:.4f}")
print(f"MSE = {mean_squared_error(y_test, y_pred):.4f}")
print(f"R2  = {r2_score(y_test, y_pred):.4f}")

# gradient descent
model_gd = LinearRegression(solver='gradient_descent', lr=0.001, epochs=5000)
model_gd.fit(X_train, y_train)
y_pred_gd = model_gd.predict(X_test)
print("\n=== Gradient Descent ===")
print(f"w = {model_gd.w}, b = {model_gd.b:.4f}")
print(f"MSE = {mean_squared_error(y_test, y_pred_gd):.4f}")
print(f"R2  = {r2_score(y_test, y_pred_gd):.4f}")
