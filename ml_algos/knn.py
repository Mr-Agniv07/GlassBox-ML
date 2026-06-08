# k-Nearest Neighbors — "lazy" learning.
# There's no training: fit() just memorizes the data. At predict time we measure
# the distance from the query point to every training point, take the k closest,
# and let them vote. Simple, but a great baseline.

import numpy as np


class KNN:
    def __init__(self, k=5):
        self.k = k

    def dist(self, x1, x2):
        # Euclidean distance.
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def fit(self, X, y):
        # Lazy: just store the data. (np.asarray so indexing/maths are safe.)
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)

    def kneighbors(self, x):
        # Indices of the k nearest training points to x — this is exactly what
        # the website lights up around the query point.
        distances = [self.dist(x, x_train) for x_train in self.X_train]
        return np.argsort(distances)[:self.k]

    def _predict(self, x):
        k_indices = self.kneighbors(x)
        k_nearest_labels = self.y_train[k_indices]

        # bincount needs non-negative ints, so cast (labels may arrive as floats).
        most_common = np.bincount(k_nearest_labels.astype(int)).argmax()
        return most_common

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._predict(x) for x in X])

    def score(self, X, y):
        # Accuracy — the shared metric for the scikit-learn comparison.
        return np.mean(self.predict(X) == np.asarray(y).astype(int))
