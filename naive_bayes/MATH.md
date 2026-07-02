# Naive Bayes

## 1. Bayes' Theorem

Naive Bayes is a probabilistic classifier built on Bayes' theorem:

$$P(y|x) = \frac{P(x|y) \, P(y)}{P(x)}$$

where:
- $P(y|x)$ — posterior probability of class $y$ given features $x$
- $P(x|y)$ — likelihood of features $x$ given class $y$
- $P(y)$ — prior probability of class $y$
- $P(x)$ — evidence (constant across classes, can be dropped)

Since $P(x)$ is the same for every class, classification reduces to:

$$\hat{y} = \arg\max_k \; P(y=k) \prod_{i=1}^{d} P(x_i|y=k)$$

## 2. The Naive Assumption

We assume all features $x_i$ are **conditionally independent** given the class $y$:

$$P(x|y) = \prod_{i=1}^{d} P(x_i|y)$$

This simplification makes estimation tractable — we only need to estimate each $P(x_i|y)$ separately rather than the full joint $P(x|y)$.

**Why this is "naive":** In practice, features are almost never independent. A word like "free" in email is correlated with "money." But classification only requires the **ranking** of posteriors to be correct, not their exact values — and the naive assumption often preserves the ranking even when it corrupts the magnitudes.

**Code:** `naive_bayes.py:32-33` — the naive assumption is encoded in how likelihoods are computed and summed:
```python
lp = self._log_gaussian(X, self.theta_[i], self.sigma_[i])   # line 32: per-feature log-likelihoods
log_post[:, i] = np.log(self.class_prior_[i]) + lp.sum(axis=1)  # line 33: sum across features = log of product
```
Line 32 computes $\log P(x_i|y=k)$ **independently per feature**. Line 33 sums them — in log-space, summation implements the product $\prod_i P(x_i|y=k)$, which is exactly the naive conditional independence assumption.

## 3. Hyperparameters

**Code:** `naive_bayes.py:5-6` — the constructor:
```python
def __init__(self, var_smoothing=1e-9):
    self.var_smoothing = var_smoothing  # line 6
```
`var_smoothing` is added to the variance estimate to prevent division by zero and numerical instability when a feature has zero variance within a class (all samples have the same value).

## 4. Gaussian Naive Bayes

For continuous features, we model each likelihood as a Gaussian:

$$P(x_i|y=k) = \frac{1}{\sqrt{2\pi\sigma_k^2}} \exp\!\left(-\frac{(x_i - \mu_k)^2}{2\sigma_k^2}\right)$$

Parameters per class $k$ and feature $i$:
- $\mu_k$ — mean of feature $i$ for class $k$
- $\sigma_k^2$ — variance of feature $i$ for class $k$

### 4.1 Parameter Estimation (MLE)

**Code:** `naive_bayes.py:8-23` — the `fit` method estimates all parameters:

```python
def fit(self, X, y):
    self.classes_ = np.unique(y)                           # line 9:  unique class labels
    n_classes = len(self.classes_)                          # line 10: number of classes K
    n_features = X.shape[1]                                # line 11: number of features d

    self.theta_ = np.zeros((n_classes, n_features))        # line 13: mu_k storage (K x d)
    self.sigma_ = np.zeros((n_classes, n_features))        # line 14: sigma_k^2 storage (K x d)
    self.class_prior_ = np.zeros(n_classes)                # line 15: P(y=k) storage (K,)

    for i, c in enumerate(self.classes_):                  # line 17: for each class k
        X_c = X[y == c]                                    # line 18: subset X with class == k
        self.theta_[i] = X_c.mean(axis=0)                  # line 19: mu_k = (1/n_k) * sum(x)
        self.sigma_[i] = X_c.var(axis=0) + self.var_smoothing  # line 20: sigma_k^2 = var(x) + epsilon
        self.class_prior_[i] = X_c.shape[0] / X.shape[0]  # line 21: P(y=k) = n_k / n
```

| Line | Formula | Description |
|------|---------|-------------|
| 19 | $\hat{\mu}_k^{(i)} = \frac{1}{n_k}\sum_{j: y_j=k} x_j^{(i)}$ | Sample mean per feature per class |
| 20 | $\hat{\sigma}_k^{2(i)} = \frac{1}{n_k}\sum_{j: y_j=k} (x_j^{(i)} - \hat{\mu}_k^{(i)})^2 + \epsilon$ | Sample variance + smoothing constant |
| 21 | $\hat{P}(y=k) = \frac{n_k}{n}$ | Class prior from training frequencies |

### 4.2 Log-Gaussian Likelihood

**Code:** `naive_bayes.py:25-27` — the log-Gaussian computation:

```python
def _log_gaussian(self, X, mean, var):                     # line 25
    return -0.5 * (np.log(2.0 * np.pi * var) + ((X - mean) ** 2) / var)  # line 27
```

This implements the log of the Gaussian PDF, dropping the $1/\sqrt{2\pi\sigma^2}$ normalization as written in log form:

$$\log P(x_i|y=k) = -\frac{1}{2}\left[\log(2\pi\sigma_k^2) + \frac{(x_i - \mu_k)^2}{\sigma_k^2}\right]$$

**Breakdown of line 27:**

| Sub-expression | Math | Purpose |
|---|---|---|
| `np.log(2.0 * np.pi * var)` | $\log(2\pi\sigma_k^2)$ | Log of the normalization constant |
| `(X - mean) ** 2` | $(x_i - \mu_k)^2$ | Squared deviation |
| `... / var` | $\frac{(x_i - \mu_k)^2}{\sigma_k^2}$ | Scaled squared deviation |
| `-0.5 * (...)` | $-\frac{1}{2}(\cdots)$ | Final log-likelihood |

`X` is `(n_samples, n_features)`, `mean` and `var` are `(n_features,)` — NumPy broadcasting handles the per-feature computation across all samples simultaneously.

## 5. Log-Space Classification

Multiplying many small probabilities causes numerical underflow. We work in log-space:

$$\log P(y=k|x) \propto \log P(y=k) + \sum_{i=1}^{d} \log P(x_i|y=k)$$

Logarithms are monotonic, so $\arg\max$ is preserved — the class with the highest log-posterior is the same as the class with the highest posterior.

### 5.1 Prediction (hard labels)

**Code:** `naive_bayes.py:29-34` — the `predict` method:

```python
def predict(self, X):                                      # line 29
    log_post = np.zeros((X.shape[0], len(self.classes_)))  # line 30: (n_samples, K) matrix
    for i, c in enumerate(self.classes_):                  # line 31: for each class k
        lp = self._log_gaussian(X, self.theta_[i], self.sigma_[i])  # line 32: log P(x|y=k) per feature
        log_post[:, i] = np.log(self.class_prior_[i]) + lp.sum(axis=1)  # line 33: log P(y=k) + sum_i log P(x_i|y=k)
    return self.classes_[np.argmax(log_post, axis=1)]      # line 34: argmax over classes
```

| Line | Formula | Description |
|------|---------|-------------|
| 30 | — | Allocate $(n, K)$ matrix for log-posteriors |
| 32 | $\log P(x_i|y=k) \;\forall\, i$ | Per-feature log-likelihoods, shape $(n, d)$ |
| 33 | $\log P(y=k) + \sum_{i=1}^d \log P(x_i\|y=k)$ | Log-prior + sum of log-likelihoods per sample |
| 34 | $\hat{y} = \arg\max_k \log P(y=k\|x)$ | Class with highest log-posterior |

### 5.2 Prediction (probabilities via softmax)

**Code:** `naive_bayes.py:36-44` — the `predict_proba` method:

```python
def predict_proba(self, X):                                # line 36
    log_post = np.zeros((X.shape[0], len(self.classes_)))  # line 37: (n_samples, K) matrix
    for i, c in enumerate(self.classes_):                  # line 38: for each class k
        lp = self._log_gaussian(X, self.theta_[i], self.sigma_[i])  # line 39: log P(x|y=k) per feature
        log_post[:, i] = np.log(self.class_prior_[i]) + lp.sum(axis=1)  # line 40: log-posterior per class
    # softmax in log-space
    max_lp = log_post.max(axis=1, keepdims=True)           # line 42: subtract max for numerical stability
    exp_post = np.exp(log_post - max_lp)                   # line 43: exponentiate shifted log-posteriors
    return exp_post / exp_post.sum(axis=1, keepdims=True)  # line 44: normalize to get probabilities
```

Lines 37–40 are identical to `predict` — they compute unnormalized log-posteriors. The difference is lines 42–44, which convert log-posteriors to proper probabilities using the **log-sum-exp trick**:

$$P(y=k|x) = \frac{\exp(\log P(y=k|x))}{\sum_{k'} \exp(\log P(y=k'|x))} = \frac{\exp(L_k - L_{\max})}{\sum_{k'} \exp(L_{k'} - L_{\max})}$$

| Line | Formula | Purpose |
|------|---------|---------|
| 42 | $L_{\max} = \max_k L_k$ | Shift to prevent overflow in exp |
| 43 | $\exp(L_k - L_{\max})$ | Exponentiate shifted values |
| 44 | $\frac{\exp(\cdot)}{\sum \exp(\cdot)}$ | Normalize to sum to 1 |

Subtracting $L_{\max}$ before exponentiating is mathematically equivalent (the shift cancels in the ratio) but prevents `np.exp` from overflowing on large log-posterior values.

## 6. Variants

| Variant | Likelihood model | Best for |
|---|---|---|
| Gaussian NB | Continuous (normal) | Real-valued features |
| Multinomial NB | Discrete counts | Text / word frequencies |
| Bernoulli NB | Binary features | Presence/absence indicators |

## Where It Breaks

See `adversarial_demo.py` for a demonstration: when classes are imbalanced, the prior term $\log P(y=k)$ at line 33 overwhelms the likelihood terms. The classifier defaults to predicting the majority class.

The prior term `np.log(self.class_prior_[i])` at line 33 is the culprit. For a 95/5 split:
- $\log P(y=0) = \log 0.95 = -0.051$
- $\log P(y=1) = \log 0.05 = -2.996$

That 2.946-nat gap must be overcome by the likelihood — but when classes overlap in feature space, the likelihoods are similar for both classes, so the prior dominates.

**Fix:** Override `class_prior_` after fitting to set equal priors, or adjust the decision threshold on `predict_proba` output.
