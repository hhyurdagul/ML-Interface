import tkinter as tk

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error


def nmse(y_true, y_pred):
    return round((((y_true - y_pred) ** 2) / (y_true.mean() * y_pred.mean())).mean(), 2)


def rmse(y_true, y_pred):
    return round(np.sqrt(mean_squared_error(y_true, y_pred)), 2)


def mae(y_true, y_pred):
    return round(mean_absolute_error(y_true, y_pred), 2)


def mape(y_true, y_pred):
    try:
        return round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 2)
    except Exception:
        return None


def smape(y_true, y_pred):
    try:
        return round(
            np.mean(np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2))
            * 100,
            2,
        )
    except Exception:
        return None


def mase(y_true, y_pred, seasons=1):
    try:
        return round(
            mean_absolute_error(y_true, y_pred)
            / mean_absolute_error(y_true[seasons:], y_true[:-seasons]),
            2,
        )
    except Exception:
        return None


skloss = {
    "NMSE": make_scorer(nmse),
    "RMSE": make_scorer(rmse),
    "MAE": make_scorer(mae),
    "MAPE": make_scorer(mape),
    "SMAPE": make_scorer(smape),
}


def loss(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> list[float]:
    NMSE = round((((y_true - y_pred) ** 2) / (y_true.mean() * y_pred.mean())).mean(), 2)
    RMSE = round(np.sqrt(mean_squared_error(y_true, y_pred)), 2)
    MAE = round(mean_absolute_error(y_true, y_pred), 2)
    try:
        MAPE = round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 2)
    except Exception:
        MAPE = None
    try:
        SMAPE = round(
            np.mean(np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2))
            * 100,
            2,
        )
    except Exception:
        SMAPE = None

    return [NMSE, RMSE, MAE, MAPE, SMAPE]

def popupmsg(msg: str) -> bool:
    tk.messagebox.showinfo("!", msg)
    return False

def cartesian(*arrays):
    mesh = np.meshgrid(*arrays)
    dim = len(mesh)
    elements = mesh[0].size
    flat = np.concatenate(mesh).ravel()
    return np.reshape(flat, (dim, elements)).T
