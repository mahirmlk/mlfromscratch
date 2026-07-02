"""
Naive KNN: explicit Python loops (no NumPy vectorization).

This is O(n*d) per query, same as the optimized version, but the
constant factor is ~100x larger because Python loops can't exploit
SIMD, cache lines, or contiguous memory access.

Compare with: knn.py (vectorized NumPy)
Run benchmark: python benchmark.py
"""

import numpy as np


class KNNNaive:
    def __init__(self, k=5, task='classification'):
        self.k = k
        self.task = task

    def fit(self, X, y):
        self.X_train = [list(row) for row in X]  # store as plain lists
        self.y_train = list(y)

    def predict(self, X):
        return np.array([self._predict_one(list(x)) for x in X])

    def _predict_one(self, x):
        # compute distances with explicit Python loops -- no numpy
        dists = []
        for i in range(len(self.X_train)):
            s = 0.0
            for j in range(len(x)):
                diff = x[j] - self.X_train[i][j]
                s += diff * diff
            dists.append((s ** 0.5, i))

        # sort by distance and take k nearest
        dists.sort(key=lambda t: t[0])
        neighbors = [self.y_train[dists[i][1]] for i in range(self.k)]

        if self.task == 'classification':
            # majority vote via dict counting
            counts = {}
            for label in neighbors:
                counts[label] = counts.get(label, 0) + 1
            return max(counts, key=counts.get)
        else:
            return sum(neighbors) / len(neighbors)
