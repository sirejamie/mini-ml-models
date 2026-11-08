# $env:PYTHONPATH=".";

import pandas as pd

from src.util.accuracy.score_r2 import calculate_r2_score
from src.util.data.data_normalization import normalize_min_max
from src.util.data.data_preparation import test_train_split
from src.regression.multiple_linear_regression import MultiLinearRegression

# ---

dataset_file = "tests/multiple_linear_regression/data/multiple_mpg_prediction_dataset.csv"
dataset = pd.read_csv(dataset_file)

X_features = dataset.drop(columns="mpg")
y_labels = dataset["mpg"]

# ---

X_train, X_test, y_train, y_test = test_train_split(X_features, y_labels)

# ---

X_train, X_test = normalize_min_max(X_train), normalize_min_max(X_test)

# ---

epochs = 1000
learning_rate = 0.05

model = MultiLinearRegression(epochs=epochs, learning_rate=learning_rate)

model.fit(X_train, y_train)
y_predictions = model.predict(X_test)

# ---

print("R2 Testing:", calculate_r2_score(y_test, y_predictions))

# ---

import numpy as np
import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(model.losses, color="red")
plt.title("Multiple Linear Regression - Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.grid(True)

count = np.arange(len(y_test))

plt.figure(2, figsize=(12, 4))
plt.scatter(count, y_predictions, color="red", label="Prediction")
plt.scatter(count, y_test, color="blue", label="Actual")
plt.title("Multiple Linear Regression - Actual vs. Predicted Plot")
plt.xlabel("Count")
plt.ylabel("Y Label")
plt.legend()
plt.grid(True)

plt.show()
