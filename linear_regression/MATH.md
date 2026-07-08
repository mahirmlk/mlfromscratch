# Linear Regression

Linear regression fits a straight line (or hyperplane) to data. You have inputs $X$ and targets $y$, and you want to find weights $w$ that make $Xw$ as close to $y$ as possible.

## The Model

$$\hat{y} = Xw + b$$

where $X \in \mathbb{R}^{n \times d}$ is your data matrix ($n$ samples, $d$ features), $w \in \mathbb{R}^d$ is the weight vector, and $b$ is a scalar bias. Typically you fold $b$ into $w$ by prepending a column of ones to $X$, so from here on we just write $Xw$.

**In the code:**
```python
# from linear_regression.py, predict() method, line 37
return X @ self.w + self.b
```
This is $\hat{y} = Xw + b$ written directly in NumPy — matrix multiply `X @ self.w` gives the linear combination, then add the scalar bias `self.b`.

**In the code (fit-time):**
```python
# from linear_regression.py, fit() method, line 27
y_pred = X @ self.w + self.b
```
Same formula computed during training to get predictions before evaluating the gradient.

## The Loss Function

We measure "how close" with mean squared error:

$$L(w) = \frac{1}{n} \|y - Xw\|^2 = \frac{1}{n} (y - Xw)^T(y - Xw)$$

Expanding that:

$$L(w) = \frac{1}{n}\left(y^Ty - 2y^TXw + w^TX^TXw\right)$$

This is a convex quadratic in $w$, so it has a single global minimum.

**In the code:** The loss is never computed explicitly — we jump straight to its gradient. But the residual vector appears directly:

```python
# from linear_regression.py, fit() method, line 28
err = y_pred - y
```
This is the vector $(Xw - y)$ that appears inside the loss. Note the sign convention: `err` is $(Xw - y)$, while the loss uses $(y - Xw)$ — they differ by a sign, but the squared norm is identical, and the gradient derivation below accounts for this.

## The Normal Equation

Set the gradient to zero and solve:

$$\nabla_w L = \frac{2}{n}\left(X^TXw - X^Ty\right) = 0$$

$$X^TXw = X^Ty$$

$$\boxed{w = (X^TX)^{-1}X^Ty}$$

This gives you the exact solution in one shot. No iteration needed. The catch: inverting $X^TX$ is $O(d^3)$, so when $d$ is large you don't want to do this.

**In the code:**
```python
# from linear_regression.py, fit() method, lines 19-22
X_b = np.hstack([np.ones((m, 1)), X])        # prepend column of 1s for bias
theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y  # (X^T X)^{-1} X^T y
self.b = theta[0]                             # extract bias (first element)
self.w = theta[1:]                            # extract weights (remaining elements)
```
Line 19 augments $X$ with a column of ones to fold $b$ into the weight vector — this is the standard trick so the normal equation solves for both $w$ and $b$ simultaneously. Line 20 is the normal equation written directly in NumPy: `X_b.T @ X_b` computes $X^TX$, `np.linalg.inv(...)` computes the inverse $(X^TX)^{-1}$, and `@ X_b.T @ y` multiplies by $X^Ty$. Lines 21-22 unpack the combined $\theta$ vector back into separate bias and weight attributes.

## Gradient Descent

Instead of solving directly, you can iteratively walk downhill:

$$w \leftarrow w - \alpha \, \nabla_w L$$

where $\alpha$ is the learning rate. Substituting the gradient:

$$w \leftarrow w - \frac{2\alpha}{n} X^T(Xw - y)$$

Repeat until convergence. Each step is $O(nd)$ — much cheaper than the normal equation when $n$ is large relative to $d$.

**In the code:**
```python
# from linear_regression.py, fit() method, lines 27-33
y_pred = X @ self.w + self.b           # predictions: Xw + b
err = y_pred - y                       # residual vector: (Xw - y)
grad_w = (2 / m) * (X.T @ err)        # gradient w.r.t. w: (2/n) X^T (Xw - y)
grad_b = (2 / m) * np.sum(err)         # gradient w.r.t. b: (2/n) sum of residuals
self.w -= self.lr * grad_w             # w <- w - alpha * grad_w
self.b -= self.lr * grad_b             # b <- b - alpha * grad_b
```
Line 27 computes predictions $\hat{y} = Xw + b$. Line 28 computes the residual $(Xw - y)$. Line 30 computes the full gradient $\frac{2}{n} X^T(Xw - y)$ in a single matrix-vector multiply — `X.T @ err` is $X^T(Xw - y)$, scaled by $\frac{2}{m}$. Line 31 computes the bias gradient $\frac{2}{n}\sum(Xw - y)$. Lines 32-33 apply the update rule with learning rate `self.lr` = $\alpha$.

## Where the Gradient Comes From

Starting from $L(w) = \frac{1}{n}(y - Xw)^T(y - Xw)$, expand and differentiate term by term:

$$\frac{\partial}{\partial w}\left[\frac{1}{n} y^Ty\right] = 0$$

$$\frac{\partial}{\partial w}\left[-\frac{2}{n} y^TXw\right] = -\frac{2}{n} X^Ty$$

$$\frac{\partial}{\partial w}\left[\frac{1}{n} w^TX^TXw\right] = \frac{2}{n} X^TXw$$

(using $\frac{\partial}{\partial w}(w^TAw) = 2Aw$ for symmetric $A$)

Stack them up:

$$\nabla_w L = \frac{2}{n}(X^TXw - X^Ty) = \frac{2}{n}X^T(Xw - y)$$

The vector $Xw - y$ is just the residual (prediction error), and $X^T$ projects it back into weight space. That's the whole gradient.

**In the code:** The gradient derivation above is implemented across three lines:

```python
# from linear_regression.py, fit() method, lines 28-30
err = y_pred - y                       # this is (Xw - y)
# the "residual" from the derivation
grad_w = (2 / m) * (X.T @ err)        # (2/n) * X^T @ (Xw - y)
# X^T projects the residual back into weight space
```
The derivation shows the gradient factors as $\frac{2}{n}X^T(Xw - y)$. The code computes this exactly: `err` holds $(Xw - y)$, `X.T @ err` applies $X^T$ to project into weight space, and `(2 / m)` scales by $\frac{2}{n}$.

## Summary

| Method | Formula | Cost | Code location |
|---|---|---|---|
| Normal equation | $w = (X^TX)^{-1}X^Ty$ | $O(d^3 + nd^2)$ | `linear_regression.py:19-22` (inside `fit()`) |
| Gradient descent | $w \leftarrow w - \frac{2\alpha}{n}X^T(Xw-y)$ | $O(nd)$ per step | `linear_regression.py:27-33` (inside `fit()`) |
| Prediction | $\hat{y} = Xw + b$ | $O(nd)$ | `linear_regression.py:37` (inside `predict()`) |

## Where It Breaks

See `adversarial_demo.py` for a demonstration: when features are multicollinear, $X^TX$ is near-singular and the normal equation explodes. The fix is Ridge regression — see [`../ridge_regression/MATH.md`](../ridge_regression/MATH.md).
