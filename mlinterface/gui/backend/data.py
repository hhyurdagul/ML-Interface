import numpy as np
import pandas as pd

from typing import Any

from mlinterface.gui.backend.scalers import MinMaxScaler, StandardScaler, ScalerBase
from mlinterface.gui.backend.utils import shift_array


class DataHandler:
    df: pd.DataFrame
    test_df: pd.DataFrame
    predictor_names: list[str]
    label_name: str

    def __read_data(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)

    def __columns(self) -> list[str]:
        if self.df is None:
            raise ValueError("Train data not loaded yet")
        return self.df.columns.tolist()

    def read_train_data(self, path: str) -> list[str]:
        self.df = self.__read_data(path)
        return self.__columns()

    def read_test_data(self, path: str) -> None:
        self.test_df = self.__read_data(path)

    def set_names(self, predictor_names: list[str], label_name: str) -> None:
        self.predictor_names = predictor_names
        self.label_name = label_name

    def get_Xy(self, train_size: int=100) -> tuple[np.ndarray, np.ndarray]:
        if self.df is None:
            raise ValueError("Train data not loaded yet")


        X = self.df[self.predictor_names].to_numpy()
        y = self.df[self.label_name].to_numpy()

        size = int((train_size / 100) * len(X))
        X, y = X[-size:], y[-size:]

        return X, y

    def get_test_Xy(self, num: int) -> tuple[np.ndarray, np.ndarray]:
        if self.test_df is None:
            raise ValueError("Test data not loaded yet")
        X = self.test_df[self.predictor_names].to_numpy()[:num]
        y = self.test_df[self.label_name].to_numpy()[:num]
        return X, y


class DataScaler:
    feature_scaler: ScalerBase | MinMaxScaler | StandardScaler
    label_scaler: ScalerBase | MinMaxScaler | StandardScaler

    def __init__(self, scale_y: bool = True) -> None:
        self.scale_y = scale_y

    def initialize(self, scaler_type: str) -> None:
        match scaler_type:
            case "Standard":
                self.feature_scaler = StandardScaler()
                self.label_scaler = StandardScaler()
            case "MinMax":
                self.feature_scaler = MinMaxScaler()
                self.label_scaler = MinMaxScaler()
            case _:
                self.feature_scaler = ScalerBase()
                self.label_scaler = ScalerBase()

        self.scaler_type = scaler_type
        self.fitted = False
        self.is_round = True
        self.is_negative = False

    def __fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.feature_scaler.fit(X)
        if self.scale_y:
            self.label_scaler.fit(y)

        self.is_round = np.issubdtype(y.dtype, np.integer)
        self.is_negative = y.min() < 0
        self.fitted = True

    def __post_process(self, y: np.ndarray) -> np.ndarray:
        y = y.astype(int) if self.is_round else y.round(2)
        y = np.clip(y, 0, np.inf) if self.is_negative else y
        return y

    def scale(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        if y.ndim != 1:
            raise ValueError("y must be a 1D array")

        if not self.fitted:
            self.__fit(X, y)

        X = self.feature_scaler.transform(X)
        if self.scale_y:
            y = self.label_scaler.transform(y.reshape(-1, 1)).ravel()

        return X, y

    def inverse_scale(self, y: np.ndarray) -> np.ndarray:
        if self.scale_y:
            y = self.label_scaler.inverse_transform(y.reshape(-1, 1)).ravel()

        return self.__post_process(y)

    def get_params(self) -> dict[str, Any]:
        data = {
            "scaler_type": self.scaler_type,
            "is_round": self.is_round,
            "is_negative": self.is_negative,
            "feature_scaler": self.feature_scaler.get_params(),
            "label_scaler": self.label_scaler.get_params(),
        }
        return data

    def set_params(self, data: dict[str, Any]):
        self.scaler_type = data["scaler_type"]
        self.is_round = data["is_round"]
        self.is_negative = data["is_negative"]
        self.feature_scaler.set_params(data["feature_scaler"])
        self.label_scaler.set_params(data["label_scaler"])

# This is a change

class LookbackHandler:
    last: np.ndarray
    merged_index: list[int]

    def initialize(
        self,
        lookback_value: int,
        seasonal_lookback_value: int,
        seasonal_lookback_period: int,
    ) -> None:
        match (seasonal_lookback_value, seasonal_lookback_period):
            case x, y if (x > 0 and y <= 0) or (x <= 0 and y > 0):
                raise ValueError(
                    "Seasonal lookback value and period both have to be zero or greater than zero"
                )

        lookback_index = list(range(1, lookback_value + 1, 1))
        seasonal_lookback_index = list(
            range(
                seasonal_lookback_period,
                (seasonal_lookback_value + 1) * seasonal_lookback_period,
                max(seasonal_lookback_period, 1),
            )
        )

        self.merged_index = list(set(lookback_index + seasonal_lookback_index))
        print(self.merged_index)

    def __get_lookback_stack(self, y: np.ndarray, n: int) -> np.ndarray:
        stack = y.copy().reshape(-1, 1)
        for i in range(1, n + 1):
            stack = np.hstack((stack, shift_array(y, i).reshape(-1, 1)))
        return stack

    def get_lookback(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.merged_index:
            stack = self.__get_lookback_stack(y, max(self.merged_index))
            stack = stack[:, self.merged_index]
            X = np.concatenate((X, stack), axis=1)

            X = X[~np.isnan(X).any(axis=1)]
            y = y[y.shape[0] - X.shape[0] :]
            self.last = y[-max(self.merged_index) :]
            self.last_to_save = self.last.copy().tolist()

        return X, y

    def append_lookback(self, X: np.ndarray) -> np.ndarray:
        if self.merged_index:
            lookback = self.last[[-i for i in self.merged_index]]
            X = np.concatenate((X.copy(), lookback))
        return X.reshape(1, -1)

    def update_last(self, value: int|float) -> None:
        if self.merged_index:
            self.last = np.append(self.last, value)[1:]

    def set_params(self, data: dict[str, Any]) -> None:
        self.merged_index = data["merged_index"]
        self.last = np.array(data["last_values"])

    def get_params(self) -> dict[str, Any]:
        data = {"last_values": self.last_to_save, "merged_index": self.merged_index}
        return data
