"""
Glassbox ML — credibility check for knn.py

Proves the from-scratch KNN matches scikit-learn's KNeighborsClassifier on the
same data, for several values of k.

Run:
    pip install numpy scikit-learn
    python test_knn.py
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from knn import KNN


def make_data(seed=0, n=300):
    """Three gaussian blobs -> 3 classes, to exercise multi-class voting."""
    rng = np.random.default_rng(seed)
    centers = [(-2, -2), (2, 2), (-2, 2)]
    X, y = [], []
    per = n // 3
    for cls, (cx, cy) in enumerate(centers):
        X.append(rng.normal((cx, cy), 0.9, size=(per, 2)))
        y.append(np.full(per, cls))
    X = np.vstack(X)
    y = np.concatenate(y).astype(float)   # floats on purpose -> tests the int cast
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def test_matches_sklearn_across_k():
    X, y = make_data()
    for k in (1, 3, 5, 11):
        mine = KNN(k=k)
        mine.fit(X, y)

        # 'brute' + uniform weights == our exact algorithm.
        sk = KNeighborsClassifier(n_neighbors=k, weights="uniform", algorithm="brute")
        sk.fit(X, y)

        pred_mine = mine.predict(X)
        pred_sk = sk.predict(X).astype(int)

        agree = np.mean(pred_mine == pred_sk)
        # Should agree almost perfectly (tiny diffs possible only on exact ties).
        assert agree > 0.98, f"k={k}: only {agree:.3f} agreement with sklearn"
        print(f"  [OK] k={k:2d}  agreement with sklearn = {agree*100:.1f}%  "
              f"(accuracy {mine.score(X, y)*100:.1f}%)")


def test_kneighbors_returns_k_indices():
    X, y = make_data()
    knn = KNN(k=7)
    knn.fit(X, y)
    nbrs = knn.kneighbors(X[0])
    assert len(nbrs) == 7
    # The closest neighbor of a point is itself (distance 0).
    assert nbrs[0] == 0
    print("  [OK] kneighbors() returns k indices, nearest is the point itself")


if __name__ == "__main__":
    print("Glassbox ML — validating knn.py against scikit-learn\n")
    test_matches_sklearn_across_k()
    test_kneighbors_returns_k_indices()
    print("\nAll checks passed. The from-scratch code matches the library.")
