"""
Glassbox ML — credibility check for logistic_regression.py

Proves the from-scratch LogisticRegression matches scikit-learn on the same
data. Evidence behind the website's "validated against scikit-learn" panel.

Run:
    pip install numpy scikit-learn
    python test_logistic_regression.py
"""

import numpy as np
from sklearn.linear_model import LogisticRegression as SkLogReg

from logistic_regression import LogisticRegression


def make_data(seed=0, n=600):
    """Two clearly separable-ish gaussian blobs -> labels 0 / 1."""
    rng = np.random.default_rng(seed)
    n0 = n // 2
    X0 = rng.normal(loc=[-1.5, -1.5], scale=1.0, size=(n0, 2))
    X1 = rng.normal(loc=[1.5, 1.5], scale=1.0, size=(n - n0, 2))
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n0), np.ones(n - n0)])
    # shuffle
    idx = rng.permutation(n)
    return X[idx], y[idx]


def test_accuracy_matches_sklearn():
    X, y = make_data()
    # GD likes standardized features.
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)

    mine = LogisticRegression(learning_rate=0.5, n_iterations=5000)
    mine.fit(X_std, y)

    # Big C ~= little regularization, so sklearn ~ plain logistic regression.
    sk = SkLogReg(C=1e6).fit(X_std, y)

    acc_mine = mine.score(X_std, y)
    acc_sk = sk.score(X_std, y)

    # Accuracies should be essentially the same.
    assert abs(acc_mine - acc_sk) < 0.01, f"acc mine={acc_mine} sk={acc_sk}"

    # The learned direction should point the same way (compare unit vectors,
    # sign included). Magnitudes differ because sklearn still regularizes a bit.
    w_mine = mine.weights / np.linalg.norm(mine.weights)
    w_sk = sk.coef_[0] / np.linalg.norm(sk.coef_[0])
    assert np.allclose(w_mine, w_sk, atol=0.05), \
        f"direction differs:\n  mine={w_mine}\n  sk  ={w_sk}"

    # The loss must actually go down over training.
    assert mine.loss_history[-1] < mine.loss_history[0]
    assert len(mine.history) == mine.n_iterations

    print(f"  [OK] Logistic regression ~= sklearn  "
          f"(acc mine={acc_mine:.3f} vs sk={acc_sk:.3f}, "
          f"log-loss {mine.loss_history[0]:.3f} -> {mine.loss_history[-1]:.3f})")


def test_predict_proba_in_range():
    X, y = make_data()
    mine = LogisticRegression(learning_rate=0.3, n_iterations=500)
    mine.fit(X, y)
    p = mine.predict_proba(X)
    assert p.min() >= 0.0 and p.max() <= 1.0
    assert set(np.unique(mine.predict(X))).issubset({0, 1})
    print("  [OK] predict_proba in [0,1], predict in {0,1}")


if __name__ == "__main__":
    print("Glassbox ML — validating logistic_regression.py against scikit-learn\n")
    test_accuracy_matches_sklearn()
    test_predict_proba_in_range()
    print("\nAll checks passed. The from-scratch code matches the library.")
