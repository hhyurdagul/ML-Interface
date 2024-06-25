import pickle
import tkinter as tk
from typing import Any

import numpy as np

def shift_array(array: np.ndarray, n: int) -> np.ndarray:
    assert array.ndim == 1, "array must be a 1D array"
    assert n >= 0, "n must be a non-negative integer"

    if n == 0:
        return array

    return np.concatenate((np.full(n, np.nan), array[:-n]))


def pickle_dump(obj: Any, file: str) -> None:
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def pickle_load(file: str) -> None:
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
