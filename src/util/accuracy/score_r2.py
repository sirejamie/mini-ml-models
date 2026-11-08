import numpy as np


def calculate_r2_score(y_actual, y_pred):
    ss_res = np.sum((y_actual - y_pred) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)

    return 1 - (ss_res / ss_tot)
