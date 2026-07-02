import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, mean_squared_error
from decision_tree import DecisionTree

np.random.seed(42)

# --- Classification ---
print("=" * 50)
print("CLASSIFICATION")
print("=" * 50)

X_cls, y_cls = make_classification(n_samples=300, n_features=5, n_informative=3,
                                    n_redundant=1, random_state=42)

# train/test split
idx = np.random.permutation(len(X_cls))
split = int(0.8 * len(X_cls))
X_train, X_test = X_cls[idx[:split]], X_cls[idx[split:]]
y_train, y_test = y_cls[idx[:split]], y_cls[idx[split:]]

clf = DecisionTree(max_depth=5, task='classification')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Tree depth: {clf.tree_depth()}")
print("\nSmall tree (first 3 levels shown by max_depth=3):")
small_clf = DecisionTree(max_depth=3, task='classification')
small_clf.fit(X_train, y_train)
small_clf.print_tree()

# --- Regression ---
print("\n" + "=" * 50)
print("REGRESSION")
print("=" * 50)

X_reg, y_reg = make_regression(n_samples=300, n_features=5, n_informative=3,
                                noise=10.0, random_state=42)

idx = np.random.permutation(len(X_reg))
split = int(0.8 * len(X_reg))
X_train, X_test = X_reg[idx[:split]], X_reg[idx[split:]]
y_train, y_test = y_reg[idx[:split]], y_reg[idx[split:]]

reg = DecisionTree(max_depth=5, task='regression')
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"Tree depth: {reg.tree_depth()}")
