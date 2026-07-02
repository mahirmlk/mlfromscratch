import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from svm import SVM

# generate linearly separable data
np.random.seed(42)
n = 100
X0 = np.random.randn(n, 2) + np.array([2, 2])
X1 = np.random.randn(n, 2) + np.array([-2, -2])
X = np.vstack([X0, X1])
y = np.array([1] * n + [-1] * n)

# fit
model = SVM(C=1.0, lr=0.001, epochs=1000)
model.fit(X, y)
print(f"Accuracy: {model.score(X, y):.2%}")
print(f"Weights: {model.w}")
print(f"Bias: {model.b:.4f}")

# plot (optional - skipped if matplotlib not available)
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='b', label='Class +1', alpha=0.6)
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c='r', label='Class -1', alpha=0.6)

    ax = plt.gca()
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], 200)
    w, b = model.w, model.b
    yy = -(w[0] * xx + b) / w[1]
    yy_margin_up = -(w[0] * xx + b - 1) / w[1]
    yy_margin_dn = -(w[0] * xx + b + 1) / w[1]
    plt.plot(xx, yy, 'k-', label='Decision boundary')
    plt.plot(xx, yy_margin_up, 'k--', alpha=0.4)
    plt.plot(xx, yy_margin_dn, 'k--', alpha=0.4)

    dec = np.abs(np.dot(X, w) + b) / np.linalg.norm(w)
    sv = dec <= 1.0 + 1e-4
    plt.scatter(X[sv, 0], X[sv, 1], s=100, linewidths=1, facecolors='none',
                edgecolors='k', label='Support vectors')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('SVM Decision Boundary')
    plt.legend()
    plt.tight_layout()
    plt.savefig('svm/decision_boundary.png', dpi=120)
    print("Saved plot to svm/decision_boundary.png")
except ImportError:
    print("matplotlib not available - skipping plot")
