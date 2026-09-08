<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/NumPy-Only-green?logo=numpy&logoColor=white" alt="NumPy Only">
  <img src="https://img.shields.io/badge/Algorithms-17-orange" alt="17 Algorithms">
  <img src="https://img.shields.io/badge/License-MIT-gray" alt="MIT License">
</p>

<h1 align="center">Machine Learning Algorithms From Scratch</h1>

<p align="center">
  <b>No frameworks. No shortcuts. Just math and NumPy.</b><br>
  Classic Machine Learning algorithms are implemented so you actually understand how they work.
</p>

---

## What's In Here

### Regression

| Algorithm | What It Does | Key Idea |
|-----------|-------------|----------|
| [Linear Regression](linear_regression/) | Fits a straight line to data | Normal equation or gradient descent |
| [Ridge Regression](ridge_regression/) | Linear regression that doesn't overfit | L2 penalty shrinks weights |
| [Lasso Regression](lasso_regression/) | Linear regression that picks features | L1 penalty zeros out weights |
| [Elastic Net](elastic_net/) | Best of Ridge + Lasso | Combined L1/L2 penalty |
| [Polynomial Regression](polynomial_regression/) | Fits curves, not just lines | Expand features, then do linear regression |
| [Bayesian Regression](bayesian_regression/) | Regression with uncertainty | Posterior over weights gives confidence intervals |

### Classification

| Algorithm | What It Does | Key Idea |
|-----------|-------------|----------|
| [Logistic Regression](logistic_regression/) | Binary classification with probabilities | Sigmoid + cross-entropy loss |
| [K Nearest Neighbors](knn/) | Classify by majority vote of neighbors | Distance-based, no training needed |
| [Naive Bayes](naive_bayes/) | Fast probabilistic classifier | Assumes features are independent |
| [Perceptron](perceptron/) | Simplest neural network | Linear boundary, SGD updates |
| [Support Vector Machine](svm/) | Maximum margin classifier | Hinge loss + SGD optimization |
| [Multilayer Perceptron](multilayer_perceptron/) | Feedforward neural network | Backpropagation through layers |

### Ensemble Methods

| Algorithm | What It Does | Key Idea |
|-----------|-------------|----------|
| [Decision Tree](decision_tree/) | Recursive binary splits | Gini impurity or entropy |
| [Random Forest](random_forest/) | Many trees, better predictions | Bagging + random feature subsets |
| [Adaboost](adaboost/) | Combine weak learners | Reweight misclassified samples |
| [Gradient Boosting](gradient_boosting/) | Sequential tree correction | Fit each tree to residuals |
| [XGBoost](xgboost/) | Regularized gradient boosting | Second-order gradients + regularization |

## How Each Algorithm Is Structured

```
algorithm_name/
├── algorithm_name.py    # The implementation (fit/predict API)
├── demo.py              # Runnable demo with synthetic or real data
└── MATH.md              # The math + how it maps to the code
```

Every class follows the same pattern:

```python
model = SomeModel(hyperparams...)
model.fit(X, y)          # train
predictions = model.predict(X)  # predict
```

## Theory Linked to Code

Each `MATH.md` doesn't just show formulas — it walks through the implementation line by line, mapping every equation to the exact code that computes it:

```markdown
### Normal Equation

$$w = (X^TX)^{-1}X^Ty$$

**In the code** (`linear_regression.py`):
```python
self.w = np.linalg.inv(X.T @ X) @ X.T @ y  # (X^T X)^{-1} X^T y
```

Every formula has a matching code block.


## Running The Demos


```bash
pip install numpy scikit-learn matplotlib

# Pick any algorithm and run its demo
python linear_regression/demo.py
python decision_tree/demo.py
python svm/demo.py
```

The demos generate data (or load small datasets), train the model, and print results. Some save plots. All of them run in under a minute.

## Dependencies

The implementations use **only NumPy**. The demos use sklearn for datasets/metrics and matplotlib for plots — but the actual models are pure math.

```
numpy          # array operations (the only real dependency)
scikit-learn   # demo-only: datasets, metrics, train_test_split
matplotlib     # demo-only: plots
```

## How To Read This Code

Start with **Linear Regression** — it's the simplest and the math carries over to everything else. Then try **Logistic Regression** to see how classification works. After that, pick whatever interests you.

Each folder has a `MATH.md` that explains the core math and walks through the code line by line. Read that first if you want to understand *why* the code does what it does.

---

<p align="center">
  <sub>Built to learn. No imports were harmed in the making of these algorithms.</sub>
</p>
