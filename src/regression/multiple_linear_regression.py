import numpy as np


class MultiLinearRegression:
    def __init__(self, epochs=1000, learning_rate=0.025):
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.weights = np.array([])
        self.bias = 0.0
        self.losses = []

    def fit(self, X, y):
        n = len(y)

        n_zeros = X.shape[1]
        self.weights = np.zeros(n_zeros)
        self.bias = 0.0
        self.losses = []

        for _ in range(self.epochs):
            mse_loss = 0.0
            weights_derivative = np.zeros(n_zeros)
            bias_derivative = 0.0

            for index, row in enumerate(X):
                y_pred = np.sum(row * self.weights) + self.bias  # y = wx + b
                err_measure = y_pred - y[index]  # (y_pred - y_actual)

                # determine mean squared error and l2 loss
                l2_loss = err_measure**2
                mse_loss += l2_loss

                # determine bias derivative
                bias_derivative += err_measure * 2

                # determine weight derivatives
                weights_derivative += err_measure * 2 * row

            # get mean of weight and bias derivative
            weights_derivative /= n
            bias_derivative /= n

            # determine mse loss
            mse_loss /= n
            self.losses.append(mse_loss)

            # final re-adjust parameters
            self.weights -= self.learning_rate * weights_derivative
            self.bias -= self.learning_rate * bias_derivative

    def predict(self, X):
        X = np.asarray(X)
        y = np.array([])

        for row in X:
            y_pred = np.sum(row * self.weights) + self.bias
            y = np.append(y, y_pred)

        return y
