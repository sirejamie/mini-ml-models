import numpy as np
import pandas as pd

dataset_file = "data/linear_regression/multiple_mpg_prediction_dataset.csv"
dataset = pd.read_csv(dataset_file)

X_features = dataset.drop(columns="mpg")
y_labels = dataset["mpg"]


def normalize_min_max(feat_matrix):
    return (feat_matrix - feat_matrix.min()) / (feat_matrix.max() - feat_matrix.min())


X_features_train = normalize_min_max(X_features[:200])
y_labels_train = y_labels[:200]

X_features_test = normalize_min_max(X_features[200:])
y_labels_test = y_labels[200:]

# ---


def fit(X, y, epochs, learning_rate):
    n = len(y)
    n_samples = X.shape[1]

    weights = np.zeros(n_samples)
    bias = 0.0

    losses = []

    for _ in range(epochs):
        mse_loss = 0.0
        weights_derivative = np.zeros(n_samples)
        bias_derivative = 0.0

        for index, row in enumerate(np.asarray(X)):
            y_pred = np.sum(row * weights) + bias  # y = wx + b
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
        losses.append(mse_loss)

        # final re-adjust parameters
        weights -= learning_rate * weights_derivative
        bias -= learning_rate * bias_derivative

    return (weights, bias), losses


# ---


def predict(X, parameters):
    weight, bias = parameters
    y = []

    for row in np.asarray(X):
        y_pred = np.sum(row * weight) + bias
        y.append(y_pred)

    return np.asarray(y)


# ---

epochs = 1500
learning_rate = 0.025

parameters, losses = fit(
    X_features_train, y_labels_train, epochs=epochs, learning_rate=learning_rate
)
y_labels_test_pred = predict(X_features_test, parameters)
y_labels_train_pred = predict(X_features_train, parameters)
# ---


def calculate_r2_score(y_actual, y_pred):
    ss_res = np.sum((y_actual - y_pred) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)

    return 1 - (ss_res / ss_tot)


print("R2 Testing: ", calculate_r2_score(y_labels_test, y_labels_test_pred))
print("R2 Training:", calculate_r2_score(y_labels_train, y_labels_train_pred))

# ---

import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(losses, color="red")
plt.title("Multiple Linear Regression - Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.grid(True)

count = np.arange(len(y_labels_test))

plt.figure(2, figsize=(12, 4))
plt.scatter(count, y_labels_test_pred, color="red", label="Prediction")
plt.scatter(count, y_labels_test, color="blue", label="Actual")
plt.title("Multiple Linear Regression - Actual vs. Predicted Plot")
plt.xlabel("Count")
plt.ylabel("Y Label")
plt.legend()
plt.grid(True)

plt.show()
