import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from adaboost import AdaBoost

np.random.seed(42)

X, y = make_classification(n_samples=200, n_features=10, n_informative=5,
                           random_state=42)
y = np.where(y == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                      random_state=42)

# Fit with default estimators
clf = AdaBoost(n_estimators=50)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
acc = np.mean(preds == y_test)
print(f"AdaBoost (50 stumps) accuracy: {acc:.4f}")

# Accuracy vs number of estimators
print("\nAccuracy vs n_estimators:")
print("-" * 35)
for n in [1, 5, 10, 20, 50]:
    clf = AdaBoost(n_estimators=n)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = np.mean(preds == y_test)
    print(f"  n={n:>3d}  accuracy={acc:.4f}")
