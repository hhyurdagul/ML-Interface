import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import dump, load  # type: ignore
from sklearn.ensemble import RandomForestRegressor

import os
import json

from .helpers import loss, popupmsg
from mlinterface.gui.backend.data import DataHandler, DataScaler, LookbackHandler
from mlinterface.gui.components.data_table import DataTable
from mlinterface.gui.components.input_component import InputComponent
from mlinterface.gui.components.model_components import ModelComponent
from mlinterface.gui.components.time_series.preprocessing_component import (
    PreprocessingComponent,
)
from mlinterface.gui.components.variables import (
    GenericIntVar,
    GenericFloatVar,
)


class RandomForest:
    def __init__(self) -> None:
        self.root = ttk.Frame()
        self.initialize()

        # Get Train Set
        self.input_component = InputComponent(
            self.root,
            text="Get Train Set",
            read_func=self.data_handler.read_train_data,
        ).grid(column=0, row=0)

        # Preprocessing Frame
        self.preprocessing_component = PreprocessingComponent(
            self.root, text="Preprocessing"
        ).grid(column=0, row=1)

        # Model
        self.model_component = ModelComponent(
            self.root,
            text="Model",
            model_type="RandomForest",
            create_model_func=self.create_model,
            save_model_func=self.save_model,
            load_model_func=self.load_model,
        ).grid(column=1, row=0)

        # Test Model
        test_model_frame = ttk.LabelFrame(self.root, text="Test Frame")
        test_model_frame.grid(column=1, row=1)

        ## Test Model Main
        test_model_main_frame = ttk.LabelFrame(test_model_frame, text="Test Model")
        test_model_main_frame.grid(column=0, row=0)

        self.prediction_count = GenericIntVar(value="")  # type: ignore
        ttk.Label(test_model_main_frame, text="Prediction Count", width=12).grid(
            column=0, row=0, sticky="w"
        )
        ttk.Entry(
            test_model_main_frame, textvariable=self.prediction_count, width=8
        ).grid(column=1, row=0, sticky="w")
        ttk.Button(
            test_model_main_frame,
            text="Result Values",
            command=self.show_result_values,
            width=8,
        ).grid(column=2, row=0, sticky="w")

        test_file_path = tk.StringVar(value="")
        ttk.Label(test_model_main_frame, text="Test File Path").grid(column=0, row=1)
        ttk.Entry(test_model_main_frame, textvariable=test_file_path).grid(
            column=1, row=1
        )
        ttk.Button(
            test_model_main_frame,
            text="Get Test Set",
            command=lambda: self.read_test_data(test_file_path),
        ).grid(column=2, row=1)

        ttk.Button(
            test_model_main_frame, text="Test Model", command=self.forecast
        ).grid(column=2, row=3)
        ttk.Button(
            test_model_main_frame,
            text="Result Graph",
            command=self.show_result_graph,
        ).grid(column=0, row=4, columnspan=3)

        ## Test Model Metrics
        test_model_metrics_frame = ttk.LabelFrame(test_model_frame, text="Test Metrics")
        test_model_metrics_frame.grid(column=1, row=0)

        test_metric_names: list[str] = ["R2", "MAE", "MAPE", "SMAPE"]
        self.test_metric_vars = [
            GenericFloatVar(value="") for _ in range(len(test_metric_names))
        ]
        for i, j in enumerate(test_metric_names):
            ttk.Label(test_model_metrics_frame, text=j, width=12).grid(column=0, row=i)
            ttk.Entry(
                test_model_metrics_frame, textvariable=self.test_metric_vars[i], width=8
            ).grid(column=1, row=i)

    def initialize(self) -> None:
        self.data_handler = DataHandler()
        self.data_scaler = DataScaler()
        self.lookback_handler = LookbackHandler()

    def read_test_data(self, file_path: tk.StringVar) -> None:
        path = filedialog.askopenfilename(
            filetypes=[
                ("Csv Files", "*.csv"),
                ("Xlsx Files", "*.xlsx"),
                ("Xlrd Files", ".xls"),
            ]
        )
        if not path:
            return
        file_path.set(path)
        self.data_handler.read_test_data(path)

    def __check_errors(self):
        try:
            self.input_component.check_errors()
            self.preprocessing_component.check_errors()

        except Exception as msg:
            popupmsg(msg)  # type: ignore
            return True

    def save_model(self):
        path = filedialog.asksaveasfilename()
        if not path:
            return
        params = {}
        try:
            model_params = self.model.get_params()
            params["n_estimators"] = model_params["n_estimators"]
            params["max_depth"] = model_params["max_depth"]
            params["min_samples_split"] = model_params["min_samples_split"]
            params["min_samples_leaf"] = model_params["min_samples_leaf"]
        except Exception:
            popupmsg("Model is not created")
            return

        params["preprocessing_component"] = self.preprocessing_component.get_params()
        params["model_component"] = self.model_component.get_params()
        params["scaler"] = self.data_scaler.get_params()

        os.mkdir(path)
        dump(self.model, path + "/model.joblib")
        with open(path + "/model.json", "w") as outfile:
            json.dump(params, outfile)

    def load_model(self):
        path = filedialog.askdirectory()
        if not path:
            return
        try:
            model_path = path + "/model.joblib"
        except Exception:
            popupmsg("There is no model file at the path")
            return
        self.model = load(model_path)
        infile = open(path + "/model.json")
        params = json.load(infile)

        self.preprocessing_component.set_params(
            params.get("preprocessing_component", {})
        )
        self.model_component.set_params(params.get("model_component", {}))

        self.data_scaler.set_params(params.get("scaler", {}))

    def __get_data(self) -> tuple[np.ndarray, np.ndarray]:
        self.data_scaler.initialize(self.preprocessing_component.scale_type.get())
        self.data_handler.set_names(
            self.input_component.get_predictors(), self.input_component.get_target()
        )

        self.lookback_handler.initialize(
            self.preprocessing_component.lookback_value.get(),
            self.preprocessing_component.seasonal_lookback_value.get(),
            self.preprocessing_component.seasonal_lookback_frequency.get(),
        )

        X, y = self.data_handler.get_Xy(
            self.preprocessing_component.train_data_size.get()
        )
        X, y = self.data_scaler.scale(X, y)
        X, y = self.lookback_handler.get_lookback(X, y)

        return X, y

    def create_model(self):
        if self.__check_errors():
            return

        X, y = self.__get_data()

        params = self.model_component.get_params()
        model = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=0,
        )
        model.fit(X, y)
        self.model = model

    def __predict(self, X_test: np.ndarray, num: int) -> np.ndarray:
        pred = []

        for i in range(num):
            in_value = self.lookback_handler.append_lookback(X_test[i])
            pred_i = self.model.predict(in_value).ravel().item()
            pred.append(pred_i)
            self.lookback_handler.update_last(pred_i)

        return np.array(pred).ravel()

    def forecast(self):
        try:
            num = self.prediction_count.get()
        except Exception:
            popupmsg("Enter a valid prediction count")
            return

        try:
            X_test, y_test = self.data_handler.get_test_Xy(num)
            self.y_test = y_test
        except Exception as msg:
            popupmsg(msg.__str__())
            return

        X_test, _ = self.data_scaler.scale(X_test, np.ones(num))
        self.pred = self.__predict(X_test, num)
        self.pred = self.data_scaler.inverse_scale(self.pred)

        losses = loss(y_test, self.pred)
        for i in range(len(self.test_metric_vars)):
            self.test_metric_vars[i].set(losses[i])

    def show_result_values(self):
        try:
            data = list(zip(self.y_test.tolist(), self.pred.tolist()))
            df = pd.DataFrame(data, columns=["Real", "Predict"])
        except Exception:
            return
        top = tk.Toplevel(self.root)
        DataTable(
            top,
            data=data,
            save_func=lambda file_path: df.to_excel(file_path, index=False),
        )
        top.mainloop()

    def show_result_graph(self):
        y_test = self.y_test
        try:
            pred = self.pred
        except Exception:
            return
        plt.plot(y_test)
        plt.plot(pred)
        plt.legend(["test", "pred"], loc="upper left")
        plt.show()
