import numpy as np

# Logistic Regression — binary classification by gradient descent.
#
# Same gradient-descent loop as LinearRegressionGD, with three swaps:
#   1. prediction passes through the sigmoid  -> a probability in (0, 1)
#   2. loss is log-loss (binary cross-entropy), not MSE
#   3. predict() thresholds the probability at 0.5 into a class {0, 1}
#
# Nice fact worth teaching on the site: even though the loss changed, the
# gradient formula  dw = (1/n) X^T (y_hat - y)  is IDENTICAL to linear
# regression. That's the signature of a "generalized linear model".


class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def sigmoid(self, z):
        # Clip z before exp() so huge magnitudes don't overflow to inf.
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Record training so the website can animate it:
        #   loss_history -> the dropping log-loss curve
        #   history      -> (weights, bias) per epoch, replayed by the scrub bar
        self.loss_history = []
        self.history = []

        for epoch in range(self.n_iterations):
            y_pred = self.sigmoid(X.dot(self.weights) + self.bias)

            dw = (1/n_samples) * X.T.dot(y_pred - y)
            db = (1/n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Binary cross-entropy. eps keeps log() away from log(0).
            eps = 1e-15
            p = np.clip(y_pred, eps, 1 - eps)
            loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
            self.loss_history.append(loss)
            self.history.append((self.weights.copy(), self.bias))

    def predict_proba(self, X):
        return self.sigmoid(X.dot(self.weights) + self.bias)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def score(self, X, y):
        # Accuracy — the shared metric used to prove this from-scratch model
        # matches scikit-learn. (R^2 doesn't apply to classification.)
        return np.mean(self.predict(X) == y)
