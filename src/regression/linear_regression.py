import numpy as np
import pandas as pd

dataset_file = "data/linear_regression/single_mpg_prediction_dataset.csv"
dataset = pd.read_csv(dataset_file)

X_features = dataset.drop(columns="mpg")
y_labels = dataset["mpg"]

# ---


def normalize_min_max(feat_matrix):
    return (feat_matrix - feat_matrix.min()) / (feat_matrix.max() - feat_matrix.min())


X_features_train = normalize_min_max(X_features[:75])
y_labels_train = y_labels[:75]

X_features_test = normalize_min_max(X_features[75:])
y_labels_test = y_labels[75:]

# ---


def fit(X, y, epochs, learning_rate):
    n = len(y)

    weight = 0.0
    bias = 0.0

    losses = []

    for _ in range(epochs):
        mse_loss = 0.0
        weight_derivative = 0.0
        bias_derivative = 0.0

        for index, row in enumerate(np.asarray(X)):
            y_pred = np.sum(row * weight) + bias  # y = wx + b
            err_measure = y_pred - y[index]  # (y_pred - y_actual)

            # determine mean squared error and l2 loss
            l2_loss = err_measure**2
            mse_loss += l2_loss

            # determine bias derivative
            bias_derivative += err_measure * 2

            # get summation of weight derivatives
            for x in row:
                weight_derivative += err_measure * 2 * x

        # get mean of weight and bias derivative
        weight_derivative /= n
        bias_derivative /= n

        # determine mse loss
        mse_loss /= n
        losses.append(mse_loss)

        # final re-adjust parameters
        weight -= learning_rate * weight_derivative
        bias -= learning_rate * bias_derivative

    return (weight, bias), losses


# ---


def predict(X, parameters):
    weight, bias = parameters
    y = []

    for row in np.asarray(X):
        y_pred = np.sum(row * weight) + bias
        y.append(y_pred)

    return np.asarray(y)


# ---

epochs = 1000
learning_rate = 0.025

parameters, losses = fit(
    X_features_train, y_labels_train, epochs=epochs, learning_rate=learning_rate
)
y_labels_pred = predict(X_features_test, parameters)

# ---


def calculate_r2_score(y_actual, y_pred):
    ss_res = np.sum((y_actual - y_pred) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)

    return 1 - (ss_res / ss_tot)


print("R2 Testing:", calculate_r2_score(y_labels_test, y_labels_pred))

# ---

import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(losses, color="red")
plt.title("Linear Regression - Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.grid(True)

count = np.arange(len(y_labels_test))

plt.figure(2, figsize=(12, 4))
plt.scatter(count, y_labels_pred, color="red", label="Prediction")
plt.scatter(count, y_labels_test, color="blue", label="Actual")
plt.title("Linear Regression - Actual vs. Predicted Plot")
plt.xlabel("Count")
plt.ylabel("Y Label")
plt.legend()
plt.grid(True)

plt.show()
