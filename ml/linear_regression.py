import numpy as np

class LinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = 0

    def fit(self, X, y, lr=0.01, epochs=1000):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        for _ in range(epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            error = y_pred - y

            # Gradient computation
            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)

            # Parameter update
            self.weights -= lr * dw
            self.bias -= lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def loss(self, X, y):
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)
