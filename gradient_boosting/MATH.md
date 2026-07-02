# Gradient Boosting — Math & Code

## Additive Model

Gradient Boosting builds an ensemble sequentially. At iteration $m$:

$$F_m(x) = F_{m-1}(x) + \eta \, h_m(x)$$

where $F_{m-1}$ is the current ensemble, $h_m$ is the new weak learner (typically a decision tree), and $\eta$ is the learning rate.

In code, each iteration adds a shrunken tree prediction to the running prediction:

```python
# line 92
pred += self.lr * tree.predict(X)
```

The learning rate (`self.lr`, set from `learning_rate`) scales the tree's contribution.

## Fitting to the Negative Gradient

Each new tree $h_m$ is fit to the **negative gradient** of the loss $L$ with respect to the current model's predictions. For training sample $i$:

$$r_{im} = -\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \Bigg|_{F = F_{m-1}}$$

These pseudo-residuals point in the direction of steepest descent in function space.

## Common Losses

### Mean Squared Error (MSE)

For $L = \frac{1}{2}(y - F)^2$ the negative gradient simplifies to ordinary residuals:

$$r_{im} = y_i - F_{m-1}(x_i)$$

This is exactly what the code computes — the residual between true labels and current predictions:

```python
# line 89
resid = y - pred
```

The tree is then fit on these residuals:

```python
# lines 90-91
tree = RegressionTree(max_depth=self.max_depth)
tree.fit(X, resid)
```

### Log-Loss (Binary Cross-Entropy)

For $L = -\bigl[y \log p + (1-y)\log(1-p)\bigr]$ with $p = \sigma(F)$:

$$r_{im} = y_i - p_i$$

*(The current implementation uses MSE loss; log-loss would require a sigmoid transform before computing residuals.)*

## Initialisation

$F_0(x) = \arg\min_c \sum_i L(y_i, c)$

For MSE loss, the optimal constant is the mean of $y$:

```python
# lines 83-84
self.init_val = np.mean(y)
pred[:] = self.init_val
```

At prediction time, the initial value is also used as the base:

```python
# line 97
pred = np.full(X.shape[0], self.init_val)
```

## Learning Rate

The learning rate $\eta \in (0, 1]$ shrinks every tree's contribution:

$$F_m(x) = F_{m-1}(x) + \eta \, h_m(x)$$

Small $\eta$ (e.g. 0.1) requires more trees but generally yields better generalization. This is the **shrinkage** regularisation strategy.

```python
# line 75
self.lr = learning_rate
```

## Loss Tracking

After each boosting iteration, MSE is recorded:

$$\text{loss}_m = \frac{1}{n} \sum_i (y_i - F_m(x_i))^2$$

```python
# line 94
self.train_losses.append(np.mean((y - pred) ** 2))
```

## Regression Tree — Split Criterion

Each weak learner is a regression tree. For a candidate split on feature $j$ at threshold $t$, the variance reduction gain is:

$$\text{gain} = n \cdot \text{Var}(y) - n_L \cdot \text{Var}(y_L) - n_R \cdot \text{Var}(y_R)$$

```python
# line 32
current_var = np.var(y) * n

# line 44
gain = current_var - np.var(y[mask]) * nl - np.var(y[~mask]) * nr
```

The best split maximizes this gain across all features and thresholds:

```python
# lines 45-48
if gain > best_gain:
    best_gain = gain
    best_feat = j
    best_thresh = t
```

### Leaf Prediction

When a node becomes a leaf (max depth reached or too few samples), its value is the mean of the residuals in that node:

$$\hat{y}_{\text{leaf}} = \frac{1}{|S|} \sum_{i \in S} r_i$$

```python
# line 27
node.val = np.mean(y)
```

## Algorithm Summary

1. **Initialise:** $F_0(x) = \arg\min_c \sum_i L(y_i, c)$
   ```python
   # line 83
   self.init_val = np.mean(y)
   ```

2. **For** $m = 1, \dots, M$:
   - Compute pseudo-residuals: $r_{im} = -\partial L / \partial F(x_i)$
     ```python
     # line 89
     resid = y - pred
     ```
   - Fit tree $h_m$ to $\{(x_i, r_{im})\}$
     ```python
     # lines 90-91
     tree = RegressionTree(max_depth=self.max_depth)
     tree.fit(X, resid)
     ```
   - Update: $F_m = F_{m-1} + \eta \, h_m$
     ```python
     # line 92
     pred += self.lr * tree.predict(X)
     ```

3. **Output:** $F_M(x)$
   ```python
   # lines 97-100
   pred = np.full(X.shape[0], self.init_val)
   for tree in self.trees:
       pred += self.lr * tree.predict(X)
   return pred
   ```
