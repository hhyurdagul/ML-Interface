import os
import pickle
from typing import List, Union, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error  # type: ignore
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate  # type: ignore
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # type: ignore
from xgboost import XGBRegressor


def handle_errors(*functions):
    for func in functions:
        if not func():
            return False
    return True


class LossHandler:
    @staticmethod
    def nmse(y_true, y_pred):
        return round(
            (((y_true - y_pred) ** 2) / (y_true.mean() * y_pred.mean())).mean(), 2
        )

    @staticmethod
    def rmse(y_true, y_pred):
        return round(np.sqrt(mean_squared_error(y_true, y_pred)), 2)

    @staticmethod
    def mae(y_true, y_pred):
        return round(mean_absolute_error(y_true, y_pred), 2)

    @staticmethod
    def mape(y_true, y_pred):
        try:
            return round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 2)
        except Exception:
            return None

    @staticmethod
    def smape(y_true, y_pred):
        try:
            return round(
                np.mean(
                    np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2)
                )
                * 100,
                2,
            )
        except Exception:
            return None

    @staticmethod
    def mase(y_true, y_pred, seasons=1):
        try:
            return round(
                mean_absolute_error(y_true, y_pred)
                / mean_absolute_error(y_true[seasons:], y_true[:-seasons]),
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

    @staticmethod
    def loss(y_true, y_pred, seasons=1):
        NMSE = round(
            (((y_true - y_pred) ** 2) / (y_true.mean() * y_pred.mean())).mean(), 2
        )
        RMSE = round(np.sqrt(mean_squared_error(y_true, y_pred)), 2)
        MAE = round(mean_absolute_error(y_true, y_pred), 2)
        try:
            MAPE = round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 2)
        except Exception:
            MAPE = None
        try:
            SMAPE = round(
                np.mean(
                    np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2)
                )
                * 100,
                2,
            )
        except Exception:
            SMAPE = None

        return [NMSE, RMSE, MAE, MAPE, SMAPE]


class ScalerHandler:
    def __init__(self) -> None:
        self.feature_scaler: Union[StandardScaler, MinMaxScaler]
        self.label_scaler: Union[StandardScaler, MinMaxScaler]
        self.scale_choice = "None"

    def set_scalers(self, scaler_type: str) -> None:
        self.scale_choice = scaler_type
        if scaler_type == "StandardScaler":
            self.feature_scaler = StandardScaler()
            self.label_scaler = StandardScaler()
        elif scaler_type == "MinMaxScaler":
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

    def scaler_fit_transform(
        self, X: pd.DataFrame, y: pd.DataFrame
    ) -> Tuple[np._typing.NDArray, np._typing.NDArray]:
        return (
            self.feature_scaler.fit_transform(X),
            self.label_scaler.fit_transform(y.values.reshape(-1, 1)).reshape(-1),
        )

    def label_inverse_transform(
        self, *data: np._typing.NDArray
    ) -> List[np._typing.NDArray]:
        return [
            self.label_scaler.inverse_transform(d.reshape(-1, 1)).reshape(-1)
            for d in data
        ]


class LookbackHandler:
    def __init__(self) -> None:
        self.last: np._typing.NDArray
        self.seasonal_last: np._typing.NDArray

    def set_variables(
        self,
        lookback: int = 0,
        seasons: int = 0,
        seasonal_lookback: int = 0,
        sliding: int = -1,
    ) -> None:
        self.lookback = lookback
        self.seasons = seasons
        self.seasonal_lookback = seasonal_lookback
        self.sliding = sliding

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

    def get_lookback(self, X, y):
        if self.sliding in [0, 2]:
            for i in range(1, self.lookback + 1):
                X[f"t-{i}"] = y.shift(i)
        elif self.sliding in [1, 2]:
            for i in range(1, self.seasons + 1):
                X[f"t-{i*self.seasonal_lookback}"] = y.shift(i * self.seasonal_lookback)

        X.dropna(inplace=True)
        a = X.to_numpy()
        b = y.iloc[-len(a) :].to_numpy().reshape(-1)

        if self.sliding in [0, 2]:
            self.set_last(b[-self.lookback :])
        elif self.sliding in [1, 2]:
            self.set_seasonal_last(b[-self.seasonal_lookback * self.seasons :])

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

    def set_variables(
        self,
        predictor_names: List[str],
        label_name: str,
        val_option: int,
        do_forecast: int,
        random_percent_value: int,
        cross_val_value: int,
    ) -> None:
        self.predictor_names = predictor_names
        self.label_name = label_name

        self.val_option = val_option
        self.do_forecast = do_forecast
        self.random_percent_value = random_percent_value
        self.cross_val_value = cross_val_value

    def get_data(self):
        X = self.train_df[self.predictor_names].copy()
        y = self.train_df[self.label_name].copy()
        return X, y

    def get_data_based_on_val_type(
        self,
        X: np._typing.NDArray,
        y: np._typing.NDArray,
    ):
        train_size = int(self.random_percent_value / 100 * len(X))
        if self.val_option == 0:
            return X, y, X, y
        elif self.val_option == 1 and not self.do_forecast:
            return train_test_split(X, y, train_size=train_size)
        elif self.val_option == 1 and self.do_forecast:
            return X[-train_size:], y[-train_size:], None, None
        else:
            return X, y, None, None


class ModelHandler:
    def __init__(
        self,
        data_handler: DataHandler,
        scaler_handler: ScalerHandler,
        lookback_handler: LookbackHandler,
    ) -> None:
        self.data_handler = data_handler
        self.scaler_handler = scaler_handler
        self.lookback_handler = lookback_handler

    def __get_data(self):
        X, y = self.data_handler.get_data()

        self.is_round = True if y.dtype in [int, np.intc, np.int64] else False
        self.is_negative = True if any(y < 0) else False

        if self.scaler_handler.scale_choice != "None":
            X.iloc[:], y.iloc[:] = self.scaler_handler.scaler_fit_transform(X, y)
        X, y = self.lookback_handler.get_lookback(X, y)
        return X, y

    def set_variables(
        self,
        model_params: dict[str, Any],
        grid_params: dict[str, Any],
        grid_option: int,
    ) -> None:
        self.model_params = model_params
        self.grid_params = grid_params
        self.grid_option = grid_option

    def create_model(self) -> None:
        X, y = self.__get_data()
        data = self.data_handler.get_data_based_on_val_type(X, y)
        X_train, y_train, X_test, y_test = data

        if not self.grid_option:
            self.model = self.__create_model_without_grid_search(self.model_params)
        else:
            self.model = self.__create_model_with_grid_search(self.grid_params)

        self.model.fit(X_train, y_train)
        if not self.data_handler.do_forecast:
            if self.data_handler.val_option in [0, 1]:
                pred = self.model.predict(X_test).reshape(-1)
                if self.scaler_handler.scale_choice != "None":
                    pred, y_test = self.scaler_handler.label_inverse_transform(
                        pred, y_test
                    )

                self.pred = pred
                self.y_test = y_test

                self.loss = LossHandler.loss(self.y_test, self.pred)

            elif self.data_handler.val_option in [2, 3] and not self.grid_option:
                cv_count = (
                    self.data_handler.cross_val_value
                    if self.data_handler.val_option == 2
                    else X_train.shape[0] - 1
                )
                cvs = cross_validate(
                    self.model, X, y, cv=cv_count, scoring=LossHandler.skloss
                )
                self.loss = [i.mean() for i in list(cvs.values())[2:]]

        if self.grid_option:
            self.model = self.model.best_estimator_
            self.best_params = self.model.get_params()

    def save_model(self) -> None:
        pass

    def load_model(self) -> None:
        pass

    def __create_model_without_grid_search(
        self, model_params: dict[str, Any]
    ) -> XGBRegressor:
        self.model = XGBRegressor(**model_params)
        return self.model

    def __create_model_with_grid_search(self, grid_params) -> GridSearchCV:
        params = {}
        interval = grid_params.get("interval")
        cv = grid_params.get("cv")

        params["n_estimators"] = np.unique(
            np.linspace(
                grid_params["n_estimators"][0],
                grid_params["n_estimators"][1],
                interval,
                dtype=int,
            )
        )
        params["max_depth"] = np.unique(
            np.linspace(
                grid_params["max_depth"][0],
                grid_params["max_depth"][1],
                interval,
                dtype=int,
            )
        )
        params["learning_rate"] = np.unique(
            np.linspace(
                grid_params["learning_rate"][0],
                grid_params["learning_rate"][1],
                interval,
                dtype=float,
            )
        )
        return GridSearchCV(XGBRegressor(), params, cv)
