import numpy as np
from mlinterface.gui.backend.scalers import MinMaxScaler, StandardScaler, ScalerBase
from mlinterface.gui.backend.utils import shift_array


class DataScaler:
    feature_scaler: ScalerBase
    label_scaler: ScalerBase

    def __init__(self, scaler_choice: str, scale_y: bool = True):
        self.scaler_choice = scaler_choice
        self.scale_y = scale_y
        match scaler_choice:
            case "StandardScaler":
                self.feature_scaler = StandardScaler()
                self.label_scaler = StandardScaler()
            case "MinMaxScaler":
                self.feature_scaler = MinMaxScaler()
                self.label_scaler = MinMaxScaler()
            case _:
                self.feature_scaler = ScalerBase()
                self.label_scaler = ScalerBase()

        self.fitted = False


    def __fit(self, X: np.ndarray, y: np.ndarray):
        self.feature_scaler.fit(X)
        if self.scale_y:
            self.label_scaler.fit(y)
        self.fitted = True

    def scale(self, X: np.ndarray, y: np.ndarray):
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

    def inverse_scale(self, y: np.array) -> np.array:
        if self.scale_y:
            y = self.label_scaler.inverse_transform(y.reshape(-1, 1)).ravel()

        return y

    def get_params(self) -> dict[str, dict[str, list[float]]]:
        data = {
            "feature_scaler": self.feature_scaler.get_params(), 
            "label_scaler": self.label_scaler.get_params()
        }
        return data

    def set_params(self, data: dict[str, dict[str, list[float]]]):
        self.feature_scaler.set_params(data["feature_scaler"])
        self.label_scaler.set_params(data["label_scaler"])


class LookbackHandler:
    def __init__(
        self,
        lookback_count: int,
        seasonal_lookback_count: int,
        seasonal_lookback_period: int,
    ):
        match (seasonal_lookback_count, seasonal_lookback_period):
            case x, y if x > 0 and y <= 0:
                raise ValueError(
                    "Seasonal lookback count and period both have to be zero or greater than zero"
                )
            case x, y if x <= 0 and y > 0:
                raise ValueError(
                    "Seasonal lookback count and period both have to be zero or greater than zero"
                )

        self.lookback_count = lookback_count
        self.seasonal_lookback_count = seasonal_lookback_count
        self.seasonal_lookback_period = seasonal_lookback_period

        lookback_index = list(range(1, lookback_count + 1, 1))
        seasonal_lookback_index = list(
            range(
                seasonal_lookback_period,
                (seasonal_lookback_count + 1) * seasonal_lookback_period,
                max(seasonal_lookback_period, 1),
            )
        )

        self.merged_index = list(
            set(lookback_index + seasonal_lookback_index)
        )

    def __get_lookback_stack(self, y: np.ndarray, n):
        stack = y.copy().reshape(-1, 1)
        for i in range(1, n + 1):
            stack = np.hstack((stack, shift_array(y, i).reshape(-1, 1)))
        return stack

    def get_lookback(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.merged_index:
            stack = self.__get_lookback_stack(y, max(self.merged_index))
            stack = stack[:, self.merged_index]
            X = np.concatenate((X, stack), axis=1)

            X = X[~np.isnan(X).any(axis=1)]
            y = y[y.shape[0] - X.shape[0] :]
            self.last = y[-max(self.merged_index):]

        return X, y

    def append_lookback(self, X: np.ndarray) -> np.ndarray:
        if self.merged_index:
            lookback = self.last[[-i for i in self.merged_index]]
            X = np.concatenate((X.copy(), lookback))
        return X.reshape(1, -1)

    def update_last(self, value: int) -> None:
        if self.merged_index:
            self.last = np.append(self.last, value)[1:]




