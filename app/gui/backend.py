import os
import pickle
from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # type: ignore


def handle_errors(*functions):
    for func in functions:
        if not func():
            return False
    return True


class ObjectHandler:
    def __init__(self) -> None:
        self.feature_scaler: Union[StandardScaler, MinMaxScaler]
        self.label_scaler: Union[StandardScaler, MinMaxScaler]
        self.last: np._typing.NDArray
        self.seasonal_last: np._typing.NDArray

    def set_scalers(self, scaler_type: str) -> None:
        if scaler_type == "StandardScaler":
            self.feature_scaler = StandardScaler()
            self.label_scaler = StandardScaler()
        else:
            self.feature_scaler = MinMaxScaler()
            self.label_scaler = MinMaxScaler()

    def set_last(self, last: np._typing.NDArray) -> None:
        self.last = last.copy()

    def set_seasonal_last(self, seasonal_last: np._typing.NDArray) -> None:
        self.seasonal_last = seasonal_last.copy()

    def save_scalers(self, path: str) -> None:
        fs_path = f"{path}/feature_scaler.pkl"
        ls_path = f"{path}/label_scaler.pkl"
        with open(fs_path, "wb") as fs, open(ls_path, "wb") as ls:
            pickle.dump(self.feature_scaler, fs)
            pickle.dump(self.label_scaler, ls)

    def load_scalers(self, path: str) -> None:
        fs_path = f"{path}/feature_scaler.pkl"
        ls_path = f"{path}/label_scaler.pkl"

        if os.path.exists(fs_path) and os.path.exists(ls_path):
            with open(fs_path, "rb") as fs, open(ls_path, "rb") as ls:
                self.feature_scaler = pickle.load(fs)
                self.label_scaler = pickle.load(ls)

    def save_lasts(self, path: str, lookback: bool, seasonal_lookback: bool) -> None:
        last_values_path = f"{path}/last_values.npy"
        slv_path = f"{path}/seasonal_last_values.npy"
        if lookback:
            np.save(last_values_path, self.last)
        if seasonal_lookback:
            np.save(slv_path, self.seasonal_last)

    def load_lasts(self, path) -> None:
        last_values_path = f"{path}/last_values.npy"
        slv_path = f"{path}/seasonal_last_values.npy"

        if os.path.exists(last_values_path):
            with open(last_values_path, "rb") as last_values:
                self.last = np.load(last_values)
        if os.path.exists(slv_path):
            with open(slv_path, "rb") as slv:
                self.seasonal_last = np.load(slv)


class DataHandler:
    def __init__(self) -> None:
        self.df: pd.DataFrame
        self.df_read = False

    def read_data(self, file_path: str) -> List[str]:
        if file_path.endswith(".csv"):
            self.df = pd.read_csv(file_path)
        else:
            try:
                self.df = pd.read_excel(file_path)
            except Exception:
                self.df = pd.read_excel(file_path, engine="openpyxl")

        self.df_read = True

        return self.df.columns.to_list()
