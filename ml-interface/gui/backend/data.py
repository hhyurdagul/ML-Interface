import numpy as np
from app.gui.backend.scalers import MinMaxScaler, StandardScaler, ScalerBase


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

    def inverse_scale(self, y):
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

