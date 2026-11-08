# $env:PYTHONPATH=".";

import pandas as pd

from src.util.accuracy.score_r2 import calculate_r2_score
from src.util.data.data_normalization import normalize_min_max
from src.util.data.data_preparation import test_train_split
from src.regression.linear_regression import LinearRegression

# ---

dataset_file = "tests/linear_regression/data/single_mpg_prediction_dataset.csv"
dataset = pd.read_csv(dataset_file)

X_features = dataset.drop(columns="mpg")
y_labels = dataset["mpg"]

# ---

X_train, X_test, y_train, y_test = test_train_split(X_features, y_labels)

# ---

X_train, X_test = normalize_min_max(X_train), normalize_min_max(X_test)

# ---

model = LinearRegression()

model.fit(X_train, y_train)
y_predictions = model.predict(X_test)

# ---

print("R2 Testing:", calculate_r2_score(y_test, y_predictions))

# ---

import numpy as np
import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(model.losses, color="red")
plt.title("Linear Regression - Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.grid(True)

count = np.arange(len(y_test))

plt.figure(2, figsize=(12, 4))
plt.scatter(count, y_predictions, color="red", label="Prediction")
plt.scatter(count, y_test, color="blue", label="Actual")
plt.title("Linear Regression - Actual vs. Predicted Plot")
plt.xlabel("Count")
plt.ylabel("Y Label")
plt.legend()
plt.grid(True)

plt.show()
