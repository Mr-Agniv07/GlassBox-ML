import numpy as np

# k-Means clustering (unsupervised — no labels).
# Repeat until the centroids stop moving:
#   1. ASSIGN  every point to its nearest centroid
#   2. UPDATE  every centroid to the mean of the points assigned to it


class KMeans:
    def __init__(self, k, max_iters=100, random_state=None):
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state

    def dist(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _assign(self, X):
        # label each point with the index of its nearest centroid
        labels = []
        for sample in X:
            distances = [self.dist(sample, c) for c in self.centroids]
            labels.append(np.argmin(distances))
        return np.array(labels)

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]

        rng = np.random.default_rng(self.random_state)
        random_idx = rng.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_idx].copy()

        # snapshots for the website animation: (centroids, labels) per step
        self.history = []

        for _ in range(self.max_iters):
            labels = self._assign(X)                       # 1. ASSIGN
            self.history.append((self.centroids.copy(), labels.copy()))

            old_centroids = self.centroids.copy()

            new_centroids = []                             # 2. UPDATE
            for cluster_id in range(self.k):
                cluster_points = X[labels == cluster_id]
                if len(cluster_points) == 0:
                    # empty cluster: keep its old centroid instead of nan
                    new_centroids.append(old_centroids[cluster_id])
                else:
                    new_centroids.append(cluster_points.mean(axis=0))
            self.centroids = np.array(new_centroids)       # set AFTER all k

            # converged once the centroids stop moving
            if np.allclose(self.centroids, old_centroids):
                break

        self.labels_ = self._assign(X)
        self.inertia_ = self._inertia(X, self.labels_)
        return self

    def _inertia(self, X, labels):
        # sum of squared distances of points to their assigned centroid
        return float(sum(self.dist(X[i], self.centroids[labels[i]]) ** 2
                         for i in range(len(X))))

    def predict(self, X):
        return self._assign(np.asarray(X, dtype=float))
