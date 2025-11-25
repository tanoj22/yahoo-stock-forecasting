# src/metrics.py

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def rmse(y_true, y_pred) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred) -> float:
    return mean_absolute_error(y_true, y_pred)

def mape(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def r2(y_true, y_pred) -> float:
    return r2_score(y_true, y_pred)
