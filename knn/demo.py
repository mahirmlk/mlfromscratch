import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from knn import KNN

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

for k in [1, 3, 5, 7, 11]:
    model = KNN(k=k)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = np.mean(preds == y_test)
    print(f"k={k:>2d}  accuracy={acc:.4f}")
