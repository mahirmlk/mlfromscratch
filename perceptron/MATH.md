# Perceptron

Every formula below is paired with the **exact line(s)** from `perceptron.py` that implement it.

---

## 1. Initialization

The weight vector and bias start at zero:

$$w = \mathbf{0} \in \mathbb{R}^n, \quad b = 0$$

**`perceptron.py` lines 13–14:**
```python
self.w = np.zeros(n_features)   # w = 0
self.b = 0                      # b = 0
```

Zero initialization means the model has no preference before seeing data — every feature starts equally weighted.

---

## 2. Forward Pass (Prediction)

Given input $x \in \mathbb{R}^n$, compute the signed activation:

$$\hat{y} = \text{sign}(w^T x + b)$$

where:

$$\text{sign}(z) = \begin{cases} +1 & \text{if } z \geq 0 \\ -1 & \text{if } z < 0 \end{cases}$$

The decision boundary is the hyperplane $w^T x + b = 0$.

**`perceptron.py` line 23:**
```python
return np.sign(np.dot(X, self.w) + self.b)
```

`np.dot(X, self.w)` computes $w^T x$ (dot product), `+ self.b` adds the bias, and `np.sign` maps the result to $\{-1, +1\}$.

---

## 3. Misclassification Check

A point $(x_i, y_i)$ is **misclassified** when:

$$y_i (w^T x_i + b) \leq 0$$

This product is negative when the prediction is wrong (signs disagree), and zero when the point lies exactly on the boundary.

**`perceptron.py` line 18:**
```python
if y[i] * (np.dot(self.w, X[i]) + self.b) <= 0:
```

`np.dot(self.w, X[i])` is $w^T x_i$, `+ self.b` adds $b$, multiplied by the true label $y_i$. If this is $\leq 0$, the update rule fires.

---

## 4. Update Rule (Perceptron Learning Rule)

When a point is misclassified, shift the boundary toward it:

$$w \leftarrow w + \eta \, y_i \, x_i$$
$$b \leftarrow b + \eta \, y_i$$

where $\eta > 0$ is the learning rate. The classic Perceptron uses $\eta = 1$; this implementation parameterizes it.

**`perceptron.py` lines 19–20:**
```python
self.w += self.lr * y[i] * X[i]   # w <- w + lr * y_i * x_i
self.b += self.lr * y[i]          # b <- b + lr * y_i
```

`self.lr` is $\eta$, `y[i]` is $y_i$, `X[i]` is $x_i$. The weight update nudges the hyperplane so this point is more likely to be classified correctly next time.

---

## 5. Training Loop

Iterate over all samples, applying the update rule to every misclassified point, and repeat for a fixed number of epochs:

$$\text{for } t = 1, \ldots, T: \quad \text{for } i = 1, \ldots, m: \quad \text{if } y_i(w^T x_i + b) \leq 0 \text{ then update}$$

**`perceptron.py` lines 16–20:**
```python
for _ in range(self.epochs):                               # outer loop: T epochs
    for i in range(n_samples):                             # inner loop: each sample
        if y[i] * (np.dot(self.w, X[i]) + self.b) <= 0:   # misclassified?
            self.w += self.lr * y[i] * X[i]                # update w
            self.b += self.lr * y[i]                       # update b
```

The outer loop (line 16) controls the maximum number of passes. The inner loop (line 17) visits every training point. The condition on line 18 gates the update.

---

## 6. Convergence Theorem

**Novikoff (1963):** If the training data is *linearly separable*, the Perceptron converges in a finite number of updates.

Let $\gamma = \min_i y_i (w^{*T} x_i + b^*) > 0$ be the margin of the optimal separator, and $R = \max_i \|x_i\|$ the maximum sample norm. Then the total number of updates is bounded by:

$$k \leq \frac{R^2}{\gamma^2}$$

**Code connection:** The `for _ in range(self.epochs)` loop on **line 16** is the only thing preventing infinite execution. If data is linearly separable, the inner loop stops updating before `epochs` is reached. If it's NOT separable, the loop just runs out — the weights oscillate and never converge.

---

## 7. Limitations — XOR Cannot Be Learned

The Perceptron cannot learn non-linearly separable functions. The classic counterexample:

| $x_1$ | $x_2$ | XOR |
|--------|--------|-----|
| 0      | 0      | -1  |
| 0      | 1      | +1  |
| 1      | 0      | +1  |
| 1      | 1      | -1  |

No single hyperplane $w^T x + b = 0$ can separate +1 from -1 here.

**Code connection — line 23:**
```python
return np.sign(np.dot(X, self.w) + self.b)
```

The prediction is $w^T x + b$ — a **linear** function of the input. No amount of training epochs can make it represent a non-linear boundary. The linearity is baked into the architecture, not the training.

---

## 8. Why MLP Fixes It

An MLP adds a hidden layer: $h = \text{relu}(W_1 x + b_1)$, then classifies in that transformed space: $\hat{y} = \text{sign}(W_2 h + b_2)$. With the right hidden representation, XOR becomes linearly separable — see [`../multilayer_perceptron/MATH.md`](../multilayer_perceptron/MATH.md).
