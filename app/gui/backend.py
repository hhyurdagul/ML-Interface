import os
import pickle
from typing import List, Union, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

from xgboost import XGBRegressor


def handle_errors(*functions):
    for func in functions:
        if not func():
            return False
    return True


class XGBModelHandler:
    def __init__(self) -> None:
        pass

    def create_model_without_grid_search(self, params: dict[str, Any]) -> XGBRegressor:
        self.model = XGBRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
        )
        return self.model



class ScalerHandler:
    def __init__(self) -> None:
        self.feature_scaler: Union[StandardScaler, MinMaxScaler]
        self.label_scaler: Union[StandardScaler, MinMaxScaler]

    def set_scalers(self, scaler_type: str) -> None:
        if scaler_type == "StandardScaler":
            self.feature_scaler = StandardScaler()
            self.label_scaler = StandardScaler()
        else:
            self.feature_scaler = MinMaxScaler()
            self.label_scaler = MinMaxScaler()

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

    def scaler_fit_transform(self, X, y):
        X = self.object_handler.feature_scaler.fit_transform(X)
        y = self.object_handler.label_scaler.fit_transform(
            y.values.reshape(-1, 1)
        ).reshape(-1)
        return X, y


class LookbackHandler:
    def __init__(self) -> None:
        self.last: np._typing.NDArray
        self.seasonal_last: np._typing.NDArray

    def set_last(self, last: np._typing.NDArray) -> None:
        self.last = last.copy()

    def set_seasonal_last(self, seasonal_last: np._typing.NDArray) -> None:
        self.seasonal_last = seasonal_last.copy()

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

    def get_lookback(
        self, X, y, lookback=0, seasons=0, seasonal_lookback=0, sliding=-1
    ):
        if sliding in [0, 2]:
            for i in range(1, lookback + 1):
                X[f"t-{i}"] = y.shift(i)
        elif sliding in [1, 2]:
            for i in range(1, seasons + 1):
                X[f"t-{i*seasonal_lookback}"] = y.shift(i * seasonal_lookback)

        X.dropna(inplace=True)
        a = X.to_numpy()
        b = y.iloc[-len(a) :].to_numpy().reshape(-1)

        if sliding in [0, 2]:
            self.set_last(b[-lookback:])
        elif sliding in [1, 2]:
            self.set_seasonal_last(b[-seasonal_lookback * seasons :])

        return a, b


class DataHandler:
    def __init__(self) -> None:
        self.train_df: pd.DataFrame
        self.train_df_read = False

        self.test_df: pd.DataFrame
        self.test_df_read = False

    def read_train_data(self, file_path: str) -> List[str]:
        if file_path.endswith(".csv"):
            self.train_df = pd.read_csv(file_path)
        else:
            self.train_df = pd.read_excel(file_path, engine="openpyxl")

        self.train_df_read = True
        return self.train_df.columns.to_list()

    def read_test_data(self, file_path: str) -> None:
        if file_path.endswith(".csv"):
            self.test_df = pd.read_csv(file_path)
        else:
            self.test_df = pd.read_excel(file_path, engine="openpyxl")

        self.test_df_read = True

    def get_data_based_on_val_type(
        self,
        X: np._typing.NDArray,
        y: np._typing.NDArray,
        val_option: int,
        do_forecast: int,
        random_percent_value: int,
    ):
        train_size = int(random_percent_value / 100 * len(X))
        if val_option == 0:
            return X, y, X, y
        elif val_option == 1 and not do_forecast:
            return train_test_split(X, y, train_size=train_size)
        elif val_option == 1 and do_forecast:
            return X[-train_size:], y[-train_size:], None, None
        else:
            return X, y, None, None
