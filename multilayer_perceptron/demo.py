import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.datasets import load_digits, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlp import MLP

# --- Binary classification ---
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                           n_classes=2, random_state=42)
X = StandardScaler().fit_transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

mlp_bin = MLP(hidden_layers=(32, 16), lr=0.01, epochs=100, batch_size=32)
mlp_bin.fit(X_tr, y_tr)
acc_bin = (mlp_bin.predict(X_te) == y_te).mean()
print(f"Binary classification accuracy: {acc_bin:.4f}")

# --- Multi-class (digits) ---
digits = load_digits()
Xd = StandardScaler().fit_transform(digits.data)
yd = digits.target
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(Xd, yd, test_size=0.2, random_state=42)

mlp_mc = MLP(hidden_layers=(128, 64), lr=0.01, epochs=100, batch_size=32)
mlp_mc.fit(Xd_tr, yd_tr)
acc_mc = (mlp_mc.predict(Xd_te) == yd_te).mean()
print(f"Digits classification accuracy: {acc_mc:.4f}")

# --- Loss curves (optional) ---
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(mlp_bin.losses)
    axes[0].set_title("Binary (make_classification)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[1].plot(mlp_mc.losses)
    axes[1].set_title("Multi-class (digits)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    plt.tight_layout()
    plt.savefig("multilayer_perceptron/mlp_loss_curves.png", dpi=100)
    print("Saved plot to multilayer_perceptron/mlp_loss_curves.png")
except ImportError:
    print("matplotlib not available - skipping plot")
