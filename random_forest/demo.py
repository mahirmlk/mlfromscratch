import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from random_forest import RandomForest, _DecisionTree

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Forest
rf = RandomForest(n_estimators=100, max_depth=10, task='clf', oob_score=True)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print(f"Random Forest accuracy: {accuracy_score(y_test, rf_preds):.4f}")
print(f"OOB score: {rf.oob_score_:.4f}")

# Single Decision Tree
tree = _DecisionTree(max_depth=10)
tree.fit(X_train, y_train)
tree_preds = tree.predict(X_test)
print(f"Single Decision Tree accuracy: {accuracy_score(y_test, tree_preds):.4f}")
