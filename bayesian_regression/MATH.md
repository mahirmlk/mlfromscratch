# Bayesian Regression 

## Model Setup

Given training data $\{X, y\}$ where $X \in \mathbb{R}^{N \times D}$ and $y \in \mathbb{R}^{N}$, Bayesian regression places a prior over the weight vector and updates it with observed data via Bayes' theorem.

### Hyperparameters

The precision hyperparameters $\alpha$ (prior) and $\beta$ (likelihood) are set at construction time:

```python
def __init__(self, alpha=1.0, beta=1.0):
    self.alpha = alpha
    self.beta = beta
```
> `alpha` controls regularization strength (prior precision); `beta` is the inverse noise variance (likelihood precision). Default both to 1.0.

### Feature Preprocessing

Inputs are standardised to zero mean and unit variance before fitting:

```python
def _preprocess(self, X, fit=False):
    if fit:
        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0) + 1e-8
    return (X - self.X_mean) / self.X_std
```
> Standardisation prevents features with large scales from dominating. The `1e-8` epsilon avoids division by zero for constant columns. Mean/std are stored during `fit` and reused during `predict`.

### Bias Column

A column of ones is prepended to incorporate the intercept term into the weight vector:

```python
def _add_bias(self, X):
    return np.hstack([np.ones((X.shape[0], 1)), X])
```
> This absorbs the bias/intercept into $w$ so the model is $y = X_b w$ where $X_b = [1, X]$.

---

## Prior Distribution

We place a zero-mean isotropic Gaussian prior on the weights:

$$w \sim \mathcal{N}(0, \alpha^{-1} I)$$

where $\alpha$ is the precision (inverse variance) hyperparameter controlling regularization strength. The prior encodes the belief that weights are small before seeing any data.

$$p(w) = \mathcal{N}(w \mid 0, \alpha^{-1} I)$$

In the code, $\alpha I$ appears as the first term of the posterior precision matrix:

```python
S_inv = self.alpha * np.eye(d) + self.beta * Xb.T @ Xb
```
> `self.alpha * np.eye(d)` implements $\alpha I$, the prior precision. A larger $\alpha$ pushes the posterior mean closer to zero (stronger regularisation).

---

## Likelihood

Assuming additive Gaussian noise with precision $\beta = \sigma^{-2}$:

$$y \mid w \sim \mathcal{N}(Xw, \beta^{-1} I)$$

$$p(y \mid X, w) = \prod_{i=1}^{N} \mathcal{N}(y_i \mid x_i^T w, \beta^{-1})$$

The data precision $\beta X^T X$ appears as the second term of the posterior precision:

```python
S_inv = self.alpha * np.eye(d) + self.beta * Xb.T @ Xb
```
> `self.beta * Xb.T @ Xb` implements $\beta X^T X$, the data contribution to precision. More data → tighter posterior.

---

## Posterior Distribution

By Bayes' theorem the posterior is proportional to likelihood times prior:

$$p(w \mid y, X) \propto p(y \mid X, w) \, p(w)$$

Since the prior and likelihood are both Gaussian, the posterior is also Gaussian (conjugacy):

$$w \mid y \sim \mathcal{N}(m_N, S_N)$$

### Posterior Covariance

$$S_N = (\alpha I + \beta X^T X)^{-1}$$

This combines the prior precision $\alpha I$ with the data precision $\beta X^T X$.

**Implementation — two lines:**

```python
S_inv = self.alpha * np.eye(d) + self.beta * Xb.T @ Xb
self.S = np.linalg.inv(S_inv)
```
> Line 1 builds the precision matrix $S_N^{-1} = \alpha I + \beta X^T X$. Line 2 inverts it to get the posterior covariance $S_N$. Note `d = Xb.shape[1]` is the number of features **plus** the bias term.

### Posterior Mean

$$m_N = \beta S_N X^T y$$

The posterior mean is a precision-weighted compromise between the prior (zero) and the maximum-likelihood solution.

```python
self.m = self.beta * self.S @ Xb.T @ y
```
> Directly implements $m_N = \beta S_N X^T y$. The matrix chain `self.S @ Xb.T @ y` computes $S_N X^T y$, scaled by $\beta$.

### Full `fit` Method

Putting it all together — preprocessing, bias addition, posterior computation:

```python
def fit(self, X, y):
    Xp = self._preprocess(X, fit=True)
    Xb = self._add_bias(Xp)
    d = Xb.shape[1]
    S_inv = self.alpha * np.eye(d) + self.beta * Xb.T @ Xb
    self.S = np.linalg.inv(S_inv)
    self.m = self.beta * self.S @ Xb.T @ y
    return self
```
> The entire posterior update is closed-form: no iterative optimisation or sampling required. This is the key advantage of conjugate Gaussian models.

---

## Predictive Distribution

For a new input $x_*$, the predictive distribution marginalises over the posterior:

$$p(y_* \mid x_*, y, X) = \int p(y_* \mid x_*, w) \, p(w \mid y, X) \, dw$$

This integral is tractable and yields:

$$y_* \mid x_*, y \sim \mathcal{N}(x_*^T m_N, \sigma_*^2)$$

### Predictive Mean

$$\hat{y}_* = x_*^T m_N$$

```python
def predict(self, X):
    Xp = self._preprocess(X)
    Xb = self._add_bias(Xp)
    return Xb @ self.m
```
> `Xb @ self.m` computes $x_*^T m_N$ for all test points simultaneously (matrix-vector product).

### Predictive Variance

$$\sigma_*^2 = \beta^{-1} + x_*^T S_N x_*$$

The first term $\beta^{-1}$ is the irreducible observation noise. The second term $x_*^T S_N x_*$ captures **epistemic uncertainty** — uncertainty due to limited training data. Far from observed data, this term grows, reflecting lower confidence.

```python
def predict_with_uncertainty(self, X):
    Xp = self._preprocess(X)
    Xb = self._add_bias(Xp)
    mu = Xb @ self.m
    var = 1.0 / self.beta + np.sum(Xb @ self.S * Xb, axis=1)
    return mu, var
```
> `1.0 / self.beta` is $\beta^{-1}$, the observation noise. `np.sum(Xb @ self.S * Xb, axis=1)` computes $x_*^T S_N x_*$ for each test point — this is the diagonal of $X_* S_N X_*^T$, using the element-wise trick to avoid forming the full $N \times N$ matrix.

---

## Key Properties

| Aspect | Detail |
|---|---|
| Regularisation | Controlled by $\alpha$; larger $\alpha$ → stronger shrinkage toward zero |
| Noise model | $\beta^{-1}$ is the assumed observation noise variance |
| Closed-form | Both posterior and predictive are Gaussian — no sampling needed |
| Uncertainty | Naturally quantified; increases away from training data |
| Preprocessing | Features are standardised (zero mean, unit variance) before fitting |
| Bias | Intercept is absorbed into the weight vector via a prepended ones column |
