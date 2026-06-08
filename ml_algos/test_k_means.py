"""
Glassbox ML — credibility check for k_means.py

k-Means is unsupervised and randomly initialized, so we don't expect identical
centroids to scikit-learn. Instead we check the things that matter:
  * it recovers the true clusters (high Adjusted Rand Index vs ground truth)
  * its solution quality (inertia) is on par with scikit-learn
  * the animation history is recorded and converges

Run:
    pip install numpy scikit-learn
    python test_k_means.py
"""

import numpy as np
from sklearn.cluster import KMeans as SkKMeans
from sklearn.metrics import adjusted_rand_score

from k_means import KMeans


def make_blobs(seed=0, per=80):
    """Four well-separated blobs -> k-means should nail them."""
    rng = np.random.default_rng(seed)
    centers = [(-6, -6), (6, 6), (-6, 6), (6, -6)]
    X, y = [], []
    for cls, (cx, cy) in enumerate(centers):
        X.append(rng.normal((cx, cy), 0.8, size=(per, 2)))
        y.append(np.full(per, cls))
    return np.vstack(X), np.concatenate(y)


def test_recovers_true_clusters():
    X, y_true = make_blobs()
    km = KMeans(k=4, max_iters=100, random_state=42).fit(X)

    ari = adjusted_rand_score(y_true, km.labels_)
    assert ari > 0.95, f"clustering doesn't match ground truth (ARI={ari:.3f})"
    print(f"  [OK] recovered true clusters  (Adjusted Rand Index = {ari:.3f})")


def test_inertia_on_par_with_sklearn():
    X, _ = make_blobs()
    km = KMeans(k=4, max_iters=100, random_state=42).fit(X)
    sk = SkKMeans(n_clusters=4, n_init=10, random_state=0).fit(X)

    # sklearn does 10 restarts and keeps the best, so it's a strong baseline.
    # On separable blobs we should be within a few percent.
    ratio = km.inertia_ / sk.inertia_
    assert ratio < 1.05, f"inertia too high vs sklearn (ratio={ratio:.3f})"

    # And our labelling should agree with sklearn's up to cluster renaming.
    ari = adjusted_rand_score(km.labels_, sk.labels_)
    assert ari > 0.95
    print(f"  [OK] inertia on par with sklearn  "
          f"(mine={km.inertia_:.1f} vs sk={sk.inertia_:.1f}, ARI vs sk={ari:.3f})")


def test_history_recorded_and_converges():
    X, _ = make_blobs()
    km = KMeans(k=4, max_iters=100, random_state=1).fit(X)
    assert len(km.history) >= 1
    # each snapshot is (centroids[k,2], labels[n])
    c0, l0 = km.history[0]
    assert c0.shape == (4, 2) and len(l0) == len(X)
    # converged before hitting the iteration cap
    assert len(km.history) < 100
    print(f"  [OK] history recorded, converged in {len(km.history)} iterations")


if __name__ == "__main__":
    print("Glassbox ML — validating k_means.py against scikit-learn\n")
    test_recovers_true_clusters()
    test_inertia_on_par_with_sklearn()
    test_history_recorded_and_converges()
    print("\nAll checks passed. The from-scratch code matches the library.")
