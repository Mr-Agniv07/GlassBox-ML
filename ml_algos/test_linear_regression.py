"""
Glassbox ML — credibility check for linear_regression.py

Proves the from-scratch implementations match scikit-learn on the same data.
This is the evidence behind the website's "validated against scikit-learn" panel.

Run:
    pip install numpy scikit-learn
    python test_linear_regression.py
"""

import numpy as np
from sklearn.linear_model import LinearRegression as SkLinReg

from linear_regression import LinearRegression, LinearRegressionGD


def make_data(seed=0, n=400, n_features=3, noise=0.4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    true_w = rng.uniform(-3, 3, size=n_features)
    true_b = 1.7
    y = X @ true_w + true_b + rng.normal(scale=noise, size=n)
    return X, y, true_w, true_b


def test_normal_equation_matches_sklearn():
    X, y, _, _ = make_data()

    mine = LinearRegression()
    mine.fit(X, y)

    sk = SkLinReg().fit(X, y)

    # Closed form is exact -> should match sklearn to ~machine precision.
    assert np.allclose(mine.weights, sk.coef_, atol=1e-8), \
        f"weights differ:\n  mine={mine.weights}\n  sk  ={sk.coef_}"
    assert np.isclose(mine.bias, sk.intercept_, atol=1e-8), \
        f"bias differs: mine={mine.bias} sk={sk.intercept_}"
    assert np.isclose(mine.score(X, y), sk.score(X, y), atol=1e-8)
    print("  [OK] Normal equation  == sklearn (exact)")


def test_gradient_descent_approaches_sklearn():
    # GD needs standardized features to converge nicely (it's scale-sensitive).
    X, y, _, _ = make_data()
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)

    mine = LinearRegressionGD(learning_rate=0.1, n_iterations=5000)
    mine.fit(X_std, y)

    sk = SkLinReg().fit(X_std, y)

    # Iterative -> won't be bit-exact, but should land very close.
    assert np.allclose(mine.weights, sk.coef_, atol=1e-2), \
        f"weights differ:\n  mine={mine.weights}\n  sk  ={sk.coef_}"
    assert np.isclose(mine.bias, sk.intercept_, atol=1e-2)
    assert np.isclose(mine.score(X_std, y), sk.score(X_std, y), atol=1e-3)

    # The loss must actually go down over training.
    assert mine.loss_history[-1] < mine.loss_history[0]
    assert len(mine.history) == mine.n_iterations
    print(f"  [OK] Gradient descent ~= sklearn  "
          f"(MSE {mine.loss_history[0]:.3f} -> {mine.loss_history[-1]:.3f})")


if __name__ == "__main__":
    print("Glassbox ML — validating linear_regression.py against scikit-learn\n")
    test_normal_equation_matches_sklearn()
    test_gradient_descent_approaches_sklearn()
    print("\nAll checks passed. The from-scratch code matches the library.")
