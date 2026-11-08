import numpy as np

class LinearRegression:
    def __init__(self, epochs=1000, learning_rate=0.025):
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.weight = 0.0
        self.bias = 0.0
        self.losses = []

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        n = len(y)

        self.weight = 0.0
        self.bias = 0.0
        self.losses = []

        for _ in range(self.epochs):
            mse_loss = 0.0
            weight_derivative = 0.0
            bias_derivative = 0.0

            for index, row in enumerate(X):
                y_pred = np.sum(row * self.weight) + self.bias  # y = wx + b
                err_measure = y_pred - y[index]  # (y_pred - y_actual)

                # determine mean squared error and l2 loss
                l2_loss = err_measure**2
                mse_loss += l2_loss

                # determine bias derivative
                bias_derivative += err_measure * 2

                # determine weight derivatives
                for x in row:
                    weight_derivative += err_measure * 2 * x

            # get mean of weight and bias derivative
            weight_derivative /= n
            bias_derivative /= n

            # determine mse loss
            mse_loss /= n
            self.losses.append(mse_loss)

            # final re-adjust parameters
            self.weight -= self.learning_rate * weight_derivative
            self.bias -= self.learning_rate * bias_derivative

    def predict(self, X):
        X = np.asarray(X)
        y = np.array([])

        for row in X:
            y_pred = np.sum(row * self.weight) + self.bias
            y = np.append(y, y_pred)

        return y