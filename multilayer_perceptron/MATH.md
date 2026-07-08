# Multilayer Perceptron

Every formula below is paired with the exact code from [`mlp.py`](mlp.py) that implements it.

---

## 1. Weight Initialization

Weights are initialized with small random values (scaled by 0.01), and biases with zeros:

$$W^{(l)}_{ij} \sim \mathcal{N}(0,\; 0.01), \qquad b^{(l)} = \mathbf{0}$$

```python
# mlp.py lines 13-15
self.W = [np.random.randn(n_in, n_out) * 0.01
          for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])]
self.b = [np.zeros((1, n_out)) for n_out in layer_sizes[1:]]
```

`np.random.randn` draws from the standard normal; multiplying by 0.01 gives small initial weights. Biases are zero-initialized per output neuron.

---

## 2. Forward Pass

For layer $l$ with weights $W^{(l)}$, biases $b^{(l)}$, and activation $\sigma$:

$$z^{(l)} = a^{(l-1)} W^{(l)} + b^{(l)}$$

$$a^{(l)} = \sigma(z^{(l)})$$

where $a^{(0)} = x$ (the input).

```python
# mlp.py lines 34-44
def _forward(self, X):
    a = X                           # a^(0) = X
    acts, zs = [a], []
    for i, (W, b) in enumerate(zip(self.W, self.b)):
        z = a @ W + b               # z^(l) = a^(l-1) @ W^(l) + b^(l)
        zs.append(z)
        if i < len(self.W) - 1:
            a = self._relu(z)       # hidden layers: ReLU activation
        else:
            a = self._softmax(z)    # output layer: softmax activation
        acts.append(a)              # store a^(l)
    return acts, zs
```

`a @ W + b` is the matrix multiplication $a^{(l-1)} W^{(l)} + b^{(l)}$. The loop applies ReLU to all hidden layers and softmax to the final output layer.

---

## 3. Activation Functions

### 3a. ReLU (hidden layers)

$$\sigma(z) = \max(0, z)$$

```python
# mlp.py lines 18-19
def _relu(z):
    return np.maximum(0, z)
```

`np.maximum(0, z)` computes element-wise $\max(0, z_{ij})$.

### 3b. ReLU Derivative

$$\sigma'(z) = \mathbf{1}(z > 0)$$

```python
# mlp.py lines 22-23
def _relu_deriv(a):
    return (a > 0).astype(float)
```

Takes the post-activation value `a`; `(a > 0)` produces a boolean mask cast to float — 1 where active, 0 where dead.

### 3c. Softmax (output layer, $K$ classes)

$$\sigma(z)_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

```python
# mlp.py lines 26-28
def _softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))  # numerical stability
    return e / e.sum(axis=1, keepdims=True)
```

Subtracting `z.max` prevents overflow in the exponentials — this shifts all values without changing the softmax result. The denominator normalizes each row to sum to 1.

### 3d. Sigmoid (available but unused in default forward pass)

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

```python
# mlp.py lines 31-32
def _sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
```

`np.clip(z, -500, 500)` guards against overflow in `np.exp`.

---

## 4. Loss Function — Cross-Entropy

For classification with one-hot targets $y$ and softmax predictions $\hat{y}$:

$$L = -\frac{1}{m} \sum_{i=1}^{m} \log \hat{y}_{i,\, y_i}$$

```python
# mlp.py lines 70-73
def _loss(self, pred, y):
    eps = 1e-12
    return -np.mean(np.log(pred[np.arange(len(y)), y] + eps))
```

`pred[np.arange(len(y)), y]` selects the predicted probability of the true class for each sample. `eps = 1e-12` prevents `log(0)`. `np.mean` divides by $m$.

---

## 5. One-Hot Encoding

Convert integer labels $y \in \{0, \dots, K-1\}$ to one-hot vectors:

$$y_k = \begin{cases} 1 & \text{if } k = y_i \\ 0 & \text{otherwise} \end{cases}$$

```python
# mlp.py lines 47-50
def _one_hot(self, y):
    oh = np.zeros((len(y), self.n_classes))
    oh[np.arange(len(y)), y] = 1
    return oh
```

Creates an $m \times K$ zero matrix, then sets position $(i, y_i)$ to 1 for each sample using advanced indexing.

---

## 6. Backpropagation

### 6a. Output error signal (softmax + cross-entropy combined gradient)

$$\delta^{(L)} = \hat{y} - y$$

```python
# mlp.py line 58
dz = acts[-1] - self._one_hot(y)
```

The well-known identity: the gradient of softmax cross-entropy simplifies to `predictions - targets`.

### 6b. Hidden layer gradient propagation

$$\delta^{(l)} = \left(\delta^{(l+1)} W^{(l+1)T}\right) \odot \sigma'(a^{(l)})$$

```python
# mlp.py line 64
dz = (dz @ self.W[i].T) * self._relu_deriv(acts[i])
```

`dz @ self.W[i].T` back-projects the error through the weight matrix. Multiplying by `_relu_deriv(acts[i])` zeros out gradients where ReLU was inactive.

### 6c. Weight gradients

$$\frac{\partial L}{\partial W^{(l)}} = \frac{1}{m} a^{(l-1)T} \delta^{(l)}$$

$$\frac{\partial L}{\partial b^{(l)}} = \frac{1}{m} \sum_{i=1}^{m} \delta^{(l)}_i$$

```python
# mlp.py lines 61-62
dW[i] = acts[i].T @ dz / m       # (1/m) * a^(l-1)^T @ delta^(l)
db[i] = dz.sum(axis=0, keepdims=True) / m   # (1/m) * sum of deltas
```

`acts[i].T @ dz` computes $a^{(l-1)T} \delta^{(l)}$; dividing by `m` averages over the batch. The bias gradient sums deltas across samples.

### 6d. Full backward loop

```python
# mlp.py lines 52-64
def _backward(self, acts, zs, y):
    m = len(y)
    n_layers = len(self.W)
    dW, db = [None] * n_layers, [None] * n_layers

    dz = acts[-1] - self._one_hot(y)           # output delta

    for i in reversed(range(n_layers)):
        dW[i] = acts[i].T @ dz / m             # weight gradient
        db[i] = dz.sum(axis=0, keepdims=True) / m  # bias gradient
        if i > 0:
            dz = (dz @ self.W[i].T) * self._relu_deriv(acts[i])  # propagate
```

The loop iterates from the output layer backward. After computing gradients at layer `i`, it propagates `dz` to layer `i-1` (unless `i == 0`, the input).

---

## 7. SGD Parameter Update

With learning rate $\eta$:

$$W^{(l)} \leftarrow W^{(l)} - \eta \frac{\partial L}{\partial W^{(l)}}$$

$$b^{(l)} \leftarrow b^{(l)} - \eta \frac{\partial L}{\partial b^{(l)}}$$

```python
# mlp.py lines 66-68
for i in range(n_layers):
    self.W[i] -= self.lr * dW[i]    # W = W - lr * dL/dW
    self.b[i] -= self.lr * db[i]    # b = b - lr * dL/db
```

`self.lr` is the learning rate $\eta$. Each parameter is updated in-place by subtracting $\eta$ times its gradient.

---

## 8. Training Loop

Mini-batch SGD with shuffled data over `epochs` iterations:

```python
# mlp.py lines 75-92
def fit(self, X, y):
    y = y.ravel().astype(int)
    self.n_classes = len(np.unique(y))
    sizes = [X.shape[1]] + list(self.hidden_layers) + [self.n_classes]
    self._init_weights(sizes)
    self.losses = []

    for _ in range(self.epochs):                           # for each epoch
        idx = np.random.permutation(len(X))                # shuffle data
        for start in range(0, len(X), self.batch_size):    # iterate mini-batches
            bi = idx[start:start + self.batch_size]
            acts, zs = self._forward(X[bi])                # forward pass
            self._backward(acts, zs, y[bi])                # backward pass + update
        acts, _ = self._forward(X)                         # full-batch loss
        self.losses.append(self._loss(acts[-1], y))
    return self
```

Each epoch shuffles indices, then processes the data in chunks of `batch_size`. After all mini-batches, the full-dataset loss is recorded.

---

## 9. Prediction

$$\hat{y} = \arg\max_k \; \sigma(z^{(L)})_k$$

```python
# mlp.py lines 94-96
def predict(self, X):
    acts, _ = self._forward(X)
    return acts[-1].argmax(axis=1)
```

`acts[-1]` is the softmax output. `argmax(axis=1)` returns the class with the highest predicted probability for each sample.
