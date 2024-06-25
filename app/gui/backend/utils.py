import tkinter as tk


import pickle
import numpy as np
from typing import Any

def shift_array(array: np.ndarray, n: int):
    assert array.ndim == 1, "array must be a 1D array"
    assert n >= 0, "n must be a non-negative integer"
    return np.concatenate((np.full(n, np.nan), array[:-n]))

def concat_X_y(X: np.ndarray, y: np.ndarray):
    assert X.ndim == 2, "X must be a 2D array"
    assert y.ndim == 1, "y must be a 1D array"
    assert X.shape[0] == y.shape[0], "X and y must have the same number of rows"

    return np.concatenate((X, y.reshape(-1, 1)), axis=1)

def pickle_dump(obj: Any, file: str):
    with open(file, "wb") as f:
        pickle.dump(obj, f)

def pickle_load(file: str):
    with open(file, "rb") as f:
        return pickle.load(f)


def cartesian(*arrays):
    mesh = np.meshgrid(*arrays)
    dim = len(mesh)
    elements = mesh[0].size
    flat = np.concatenate(mesh).ravel()
    return np.reshape(flat, (dim, elements)).T

def popupmsg(msg):
    tk.messagebox.showinfo("!", msg)
    return False

