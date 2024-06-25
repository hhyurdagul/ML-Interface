import tkinter as tk

import numpy as np
from sklearn.metrics import make_scorer, mean_absolute_error, root_mean_squared_error

def shift_array(array: np.ndarray, n: int):
    assert array.ndim == 1, "array must be a 1D array"
    assert n >= 0, "n must be a non-negative integer"
    return np.concatenate((np.full(n, np.nan), array[:-n]))

def concat_X_y(X: np.ndarray, y: np.ndarray):
    assert X.ndim == 2, "X must be a 2D array"
    assert y.ndim == 1, "y must be a 1D array"
    assert X.shape[0] == y.shape[0], "X and y must have the same number of rows"

    return np.concatenate((X, y.reshape(-1, 1)), axis=1)

def cartesian(*arrays):
    mesh = np.meshgrid(*arrays)
    dim = len(mesh)
    elements = mesh[0].size
    flat = np.concatenate(mesh).ravel()
    return np.reshape(flat, (dim, elements)).T

def popupmsg(msg):
    tk.messagebox.showinfo("!", msg)
    return False

