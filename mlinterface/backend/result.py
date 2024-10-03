from functools import wraps

import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)

def round_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs) -> float:
        return round(func(*args, **kwargs), 2)

    return wrapper


class RegressionResult:
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        if y_true.ndim != 1 or y_pred.ndim != 1:
            raise ValueError("y_true and y_pred must be 1D arrays")

        self.y_true = y_true
        self.y_pred = y_pred

    @property
    @round_wrapper
    def r2(self) -> float:
        r = r2_score(self.y_true, self.y_pred)
        if isinstance(r, np.ndarray):
            return np.mean(r, 0).item()
        else:
            return r

    @property
    @round_wrapper
    def nmse(self) -> float:
        numerator = ((self.y_true - self.y_pred) ** 2).mean()
        denominator = self.y_true.mean() * self.y_pred.mean()
        return (numerator / denominator).item()

    @property
    @round_wrapper
    def rmse(self) -> float:
        return root_mean_squared_error(self.y_true, self.y_pred).item()

    @property
    @round_wrapper
    def mae(self) -> float:
        return mean_absolute_error(self.y_true, self.y_pred).item()

    @property
    @round_wrapper
    def mape(self) -> float:
        return mean_absolute_percentage_error(self.y_true, self.y_pred).item() * 100

    @property
    @round_wrapper
    def smape(self) -> float:
        numerator = np.abs(self.y_true - self.y_pred)
        denominator = (np.abs(self.y_true) + np.abs(self.y_pred)) / 2
        return np.mean(numerator / denominator).item() * 100

    def __call__(self) -> list[float]:
        return [
            self.r2,
            self.mae,
            self.mape,
            self.smape,
        ]

