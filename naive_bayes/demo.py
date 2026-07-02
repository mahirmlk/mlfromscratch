import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB as SklearnNB
from sklearn.metrics import accuracy_score

from naive_bayes import GaussianNB

# load data
X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

# our implementation
ours = GaussianNB()
ours.fit(X_tr, y_tr)
y_pred_ours = ours.predict(X_te)

# sklearn
sk = SklearnNB()
sk.fit(X_tr, y_tr)
y_pred_sk = sk.predict(X_te)

print(f"Our NB accuracy:     {accuracy_score(y_te, y_pred_ours):.4f}")
print(f"Sklearn NB accuracy: {accuracy_score(y_te, y_pred_sk):.4f}")
print(f"Predictions match:   {np.all(y_pred_ours == y_pred_sk)}")
