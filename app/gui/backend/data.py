import pickle
import numpy as np
import pandas as pd
from typing import Tuple, TypeAlias, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer

def pickle_dump(obj: Any, file: str):
    with open(file, "wb") as f:
        pickle.dump(obj, f)

def pickle_load(file: str):
    with open(file, "rb") as f:
        return pickle.load(f)

class DataScaler:
    def __init__(self, scaler_choice: str):
        self.scaler_choice = scaler_choice
        if scaler_choice == "StandardScaler":
            self.feature_scaler = StandardScaler()
            self.label_scaler = StandardScaler()
        elif scaler_choice == "MinMaxScaler":
            self.feature_scaler = MinMaxScaler()
            self.label_scaler = MinMaxScaler()
        self.fitted_X = False
        self.fitted_y = False

    def __fit_X(self, X: pd.DataFrame):
        if not self.fitted_X:
            self.feature_scaler.fit(X)
            self.fitted = True

    def __fit_y(self, y: pd.Series):
        if not self.fitted_y:
            self.label_scaler.fit(y)
            self.fitted = True

    def scale_X(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.scaler_choice == "None":
            return X
        self.__fit_X(X)
        return pd.DataFrame(data=self.feature_scaler.transform(X), columns=X.columns)

    def unscale_X(self, X: np.ndarray) -> np.ndarray:
        if self.scaler_choice == "None":
            return X
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        return self.feature_scaler.inverse_transform(X)

    def scale_y(self, y: pd.Series) -> pd.Series:
        if self.scaler_choice == "None":
            return y
        self.__fit_y(y)
        return pd.Series(
            data=self.label_scaler.transform(y.to_numpy().reshape(-1, 1)).ravel(),
            name=y.name,
        )

    def unscale_y(self, y: np.ndarray) -> np.ndarray:
        if self.scaler_choice == "None":
            return y
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        return self.label_scaler.inverse_transform(y.reshape(-1, 1)).ravel()

    def save_scalers(self, path: str) -> None:
        if self.scaler_choice == "None":
            return 
        pickle_dump(self.feature_scaler, path + "/feature_scaler.pickle")
        pickle_dump(self.label_scaler, path + "/label_scaler.pickle")
        # NOTE: Migration check the scaler paths were ending with .pkl not .pickle

    def load_scalers(self, path: str) -> None:
        if self.scaler_choice == "None":
            return 
        self.feature_scaler = pickle_load(path + "/feature_scaler.pickle")
        self.label_scaler = pickle_load(path + "/label_scaler.pickle")
        self.fitted_X = True
        self.fitted_y = True

