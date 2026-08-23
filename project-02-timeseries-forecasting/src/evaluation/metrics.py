"""
Shared forecast evaluation metrics.

Author: Manuel Corona
"""

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_forecast(y_true, y_pred) -> Dict[str, float]:
    """
    Compute MAE, RMSE, and MAPE for a set of forecasts.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values, same length/order as y_true.

    Returns:
        Dict with mae, rmse, mape (mape as a percentage).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"mae": mae, "rmse": rmse, "mape": mape}


def print_metrics(name: str, metrics: Dict[str, float]):
    print(f"{name:25s} MAE={metrics['mae']:8.2f}  RMSE={metrics['rmse']:8.2f}  MAPE={metrics['mape']:6.2f}%")
