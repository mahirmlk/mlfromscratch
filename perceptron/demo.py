import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron

# generate linearly separable data
np.random.seed(42)
n = 100
X1 = np.random.randn(n // 2, 2) + np.array([2, 2])
X2 = np.random.randn(n // 2, 2) + np.array([-2, -2])
X = np.vstack([X1, X2])
y = np.array([1] * (n // 2) + [-1] * (n // 2))

# fit and predict
p = Perceptron(lr=0.01, epochs=1000)
p.fit(X, y)
y_pred = p.predict(X)

acc = np.mean(y_pred == y)
print(f"Accuracy: {acc:.2%}")

# plot decision boundary
fig, ax = plt.subplots()
ax.scatter(X[y == 1, 0], X[y == 1, 1], c='b', label='Class 1')
ax.scatter(X[y == -1, 0], X[y == -1, 1], c='r', label='Class -1')

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
xx = np.linspace(x_min, x_max, 100)
if p.w[1] != 0:
    yy = -(p.w[0] * xx + p.b) / p.w[1]
    ax.plot(xx, yy, 'k--', label='Decision boundary')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.legend()
ax.set_title('Perceptron Classification')
plt.tight_layout()
plt.savefig('perceptron/decision_boundary.png', dpi=150)
print("Plot saved to perceptron/decision_boundary.png")
plt.show()
