import tkinter as tk
from tkinter import ttk

import numpy as np
from sklearn.metrics import (make_scorer, mean_absolute_error,
                             mean_squared_error)


def nmse(y_true, y_pred):
    return round(
        (((y_true - y_pred)**2) / (y_true.mean() * y_pred.mean())).mean(), 2)


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
            np.mean(
                np.abs(y_true - y_pred) /
                ((np.abs(y_true) + np.abs(y_pred)) / 2)) * 100,
            2,
        )
    except Exception:
        return None


def mase(y_true, y_pred, seasons=1):
    try:
        return round(
            mean_absolute_error(y_true, y_pred) /
            mean_absolute_error(y_true[seasons:], y_true[:-seasons]),
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


def loss(y_true, y_pred, seasons=1):
    NMSE = round(
        (((y_true - y_pred)**2) / (y_true.mean() * y_pred.mean())).mean(), 2)
    RMSE = round(np.sqrt(mean_squared_error(y_true, y_pred)), 2)
    MAE = round(mean_absolute_error(y_true, y_pred), 2)
    try:
        MAPE = round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 2)
    except Exception:
        MAPE = None
    try:
        SMAPE = round(
            np.mean(
                np.abs(y_true - y_pred) /
                ((np.abs(y_true) + np.abs(y_pred)) / 2)) * 100,
            2,
        )
    except Exception:
        SMAPE = None

    return [NMSE, RMSE, MAE, MAPE, SMAPE]


def popupmsg(msg):
    popup = tk.Tk()
    popup.wm_title("!")
    label = ttk.Label(popup, text=msg)
    label.pack(side="top", fill="x", pady=10)
    B1 = ttk.Button(popup, text="Okay", command=popup.destroy)
    B1.pack()
    popup.mainloop()


def waitmsg(msg, size=100):
    popup = tk.Tk()
    popup.wm_title("!")
    ttk.Label(popup, text=msg).pack()
    pb = ttk.Progressbar(popup, length=size, mode="indeterminate")
    pb.pack()
    pb.start()
    popup.update()
    return popup


def cartesian(*arrays):
    mesh = np.meshgrid(*arrays)
    dim = len(mesh)
    elements = mesh[0].size
    flat = np.concatenate(mesh).ravel()
    return np.reshape(flat, (dim, elements)).T
