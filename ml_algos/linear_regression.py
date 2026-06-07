
# WITHOUT GRADIENT DESCENT

import numpy as np

# Direct Normal Equation Solution for Linear Regression

#Normal equation: theta = (X^T * X)^(-1) * X^T * y works fine for small datasets but can be computationally expensive for large datasets due to the matrix inversion step. It is not recommended for large datasets or when the number of features is very high, as it can lead to numerical instability and increased computational time.
# In practice, for large datasets, it is often more efficient to use iterative optimization algorithms like Gradient Descent or Stochastic Gradient Descent, which can handle larger datasets without the need for matrix inversion and can converge to the optimal solution more efficiently.

class LinearRegression:

    def fit(self, X, y):

        self.X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # Solve the system instead of inverting the matrix: np.linalg.solve is
        # numerically stabler and faster than inv(...).dot(...). This avoids the
        # exact instability the note above warns about. (pinv would also handle
        # a singular X^T X, e.g. perfectly correlated features.)
        self.theta = np.linalg.solve(self.X_b.T @ self.X_b, self.X_b.T @ y)

        self.bias = self.theta[0]
        self.weights = self.theta[1:]

    def predict(self, X):
        return self.bias + X.dot(self.weights)

    def score(self, X, y):
        # R^2 (coefficient of determination) — the shared metric used to prove
        # this from-scratch model matches scikit-learn.
        ss_res = np.sum((y - self.predict(X)) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot


# With Gradient Descent

class LinearRegressionGD:

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Record training so the website can animate it:
        #   loss_history -> the dropping MSE curve
        #   history      -> (weights, bias) per epoch, replayed by the scrub bar
        self.loss_history = []
        self.history = []

        for epoch in range(self.n_iterations):
            y_pred = X.dot(self.weights) + self.bias

            dw = (2/n_samples) * X.T.dot(y_pred - y)
            db = (2/n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            mse = np.mean((y_pred - y) ** 2)
            self.loss_history.append(mse)
            self.history.append((self.weights.copy(), self.bias))

    def predict(self, X):
        return X.dot(self.weights) + self.bias

    def score(self, X, y):
        ss_res = np.sum((y - self.predict(X)) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot

