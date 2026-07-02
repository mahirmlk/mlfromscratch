import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from bayesian_regression import BayesianRegression

np.random.seed(42)

# generate data: sin curve + noise
X = np.linspace(0, 2 * np.pi, 50).reshape(-1, 1)
y = np.sin(X).ravel() + 0.2 * np.random.randn(50)

# fit
model = BayesianRegression(alpha=0.5, beta=5.0)
model.fit(X, y)

# predict on dense grid
X_test = np.linspace(-1, 7, 200).reshape(-1, 1)
mu, var = model.predict_with_uncertainty(X_test)
std = np.sqrt(var)

print(f"Posterior mean weights: {model.m}")
print(f"Prediction range: [{mu.min():.3f}, {mu.max():.3f}]")
print(f"Avg uncertainty (std): {std.mean():.3f}")

# plot (optional)
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.scatter(X, y, c='k', s=20, label='data', zorder=3)
    plt.plot(X_test, mu, 'b-', label='mean prediction')
    plt.fill_between(X_test.ravel(), mu - 2 * std, mu + 2 * std,
                     alpha=0.3, color='blue', label=r'$\pm 2\sigma$')
    plt.plot(X_test, np.sin(X_test), 'r--', alpha=0.5, label='true sin(x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Bayesian Regression')
    plt.legend()
    plt.tight_layout()
    plt.savefig('bayesian_regression/demo_plot.png', dpi=120)
    print('Plot saved to bayesian_regression/demo_plot.png')
except ImportError:
    print("matplotlib not available - skipping plot")
