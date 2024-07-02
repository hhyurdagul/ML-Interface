import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, cross_validate

import os
import json

from .helpers import loss, skloss, popupmsg
from mlinterface.gui.backend.data import DataHandler, DataScaler, LookbackHandler
from mlinterface.gui.components.data_table import DataTable
from mlinterface.gui.components.input_component import InputComponent
from mlinterface.gui.components.variables import GenericVar

from typing import Tuple


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

        # Model testing and validation
        model_validation_frame = ttk.Labelframe(
            self.root, text="Model testing and validation"
        )
        model_validation_frame.grid(column=0, row=1)

        self.do_forecast_option = GenericVar(value=0)
        tk.Checkbutton(
            model_validation_frame,
            text="Do Forecast",
            offvalue=0,
            onvalue=1,
            variable=self.do_forecast_option,
            command=self.__open_other_entries,
        ).grid(column=0, row=0, columnspan=2)

        self.validation_option = GenericVar(value=0)
        self.random_percent_var = GenericVar(value=70)
        self.cross_val_var = GenericVar(value=5)
        tk.Radiobutton(
            model_validation_frame,
            text="No validation, use all data rows",
            value=0,
            variable=self.validation_option,
            command=self.__open_other_entries,
        ).grid(column=0, row=1, columnspan=2, sticky=tk.W)
        tk.Radiobutton(
            model_validation_frame,
            text="Random percent",
            value=1,
            variable=self.validation_option,
            command=self.__open_other_entries,
        ).grid(column=0, row=2, sticky=tk.W)
        self.cv_entry_1 = tk.Radiobutton(
            model_validation_frame,
            text="K-fold cross-validation",
            value=2,
            variable=self.validation_option,
            command=self.__open_other_entries,
        )
        self.cv_entry_1.grid(column=0, row=3, sticky=tk.W)
        self.cv_entry_2 = tk.Radiobutton(
            model_validation_frame,
            text="Leave one out cross-validation",
            value=3,
            variable=self.validation_option,
            command=self.__open_other_entries,
        )
        self.cv_entry_2.grid(column=0, row=4, columnspan=2, sticky=tk.W)
        self.random_percent_entry = ttk.Entry(
            model_validation_frame, textvariable=self.random_percent_var, width=8
        )
        self.random_percent_entry.grid(column=1, row=2)
        self.cv_value_entry = ttk.Entry(
            model_validation_frame, textvariable=self.cross_val_var, width=8
        )
        self.cv_value_entry.grid(column=1, row=3)

        # Customize Train Set
        customize_train_set_frame = ttk.LabelFrame(
            self.root, text="Customize Train Set"
        )
        customize_train_set_frame.grid(column=0, row=2)

        self.lookback_option = GenericVar(value=0)
        self.lookback_val_var = GenericVar(value="")  # type: ignore
        tk.Checkbutton(
            customize_train_set_frame,
            text="Lookback",
            offvalue=0,
            onvalue=1,
            variable=self.lookback_option,
            command=self.__open_other_entries,
        ).grid(column=0, row=0)
        self.lookback_entry = tk.Entry(
            customize_train_set_frame,
            textvariable=self.lookback_val_var,
            width=8,
            state=tk.DISABLED,
        )
        self.lookback_entry.grid(column=1, row=0)

        self.seasonal_lookback_option = GenericVar(value=0)
        self.seasonal_val_var = GenericVar(value="")
        self.seasonal_period_var = GenericVar(value="")
        tk.Checkbutton(
            customize_train_set_frame,
            text="Periodic Lookback",
            offvalue=0,
            onvalue=1,
            variable=self.seasonal_lookback_option,
            command=self.__open_other_entries,
        ).grid(column=0, row=1)
        self.seasonal_lookback_entry_1 = tk.Entry(
            customize_train_set_frame,
            textvariable=self.seasonal_period_var,
            width=9,
            state=tk.DISABLED,
        )
        self.seasonal_lookback_entry_1.grid(column=0, row=2)
        self.seasonal_lookback_entry_2 = tk.Entry(
            customize_train_set_frame,
            textvariable=self.seasonal_val_var,
            width=8,
            state=tk.DISABLED,
        )
        self.seasonal_lookback_entry_2.grid(column=1, row=2)

        self.scale_var = tk.StringVar(value="None")
        ttk.Label(customize_train_set_frame, text="Scale Type").grid(column=0, row=3)
        ttk.OptionMenu(
            customize_train_set_frame,
            self.scale_var,
            "None",
            "None",
            "StandardScaler",
            "MinMaxScaler",
        ).grid(column=1, row=3)

        # Model
        model_frame = ttk.Labelframe(self.root, text="Model Frame")
        model_frame.grid(column=1, row=0)

        ## Parameter Optimization
        parameter_optimization_frame = ttk.Labelframe(
            model_frame, text="Parameter Optimization"
        )
        parameter_optimization_frame.grid(column=0, row=2)

        self.grid_option_var = GenericVar(value=0)
        tk.Checkbutton(
            parameter_optimization_frame,
            text="Do grid search for optimal parameters",
            offvalue=0,
            onvalue=1,
            variable=self.grid_option_var,
            command=self.__open_entries,
        ).grid(column=0, row=0, columnspan=3)

        self.interval_var = GenericVar(value=3)
        ttk.Label(parameter_optimization_frame, text="Interval:").grid(column=0, row=1)
        self.interval_entry = ttk.Entry(
            parameter_optimization_frame,
            textvariable=self.interval_var,
            width=8,
            state=tk.DISABLED,
        )
        self.interval_entry.grid(column=1, row=1, pady=2)

        self.gs_cross_val_option = GenericVar(value=0)
        self.gs_cross_val_var = GenericVar(value=5)
        tk.Checkbutton(
            parameter_optimization_frame,
            text="Cross validate; folds:",
            offvalue=0,
            onvalue=1,
            variable=self.gs_cross_val_option,
            command=self.__open_entries,
        ).grid(column=0, row=2)
        self.gs_cross_val_entry = tk.Entry(
            parameter_optimization_frame,
            textvariable=self.gs_cross_val_var,
            state=tk.DISABLED,
            width=8,
        )
        self.gs_cross_val_entry.grid(column=1, row=2)

        ## Model Parameters
        model_parameters_frame = ttk.LabelFrame(model_frame, text="Model Parameters")
        model_parameters_frame.grid(column=1, row=0, rowspan=3, columnspan=2)

        parameter_names = [
            "N Estimators",
            "Max Depth",
            "Min Samples Split",
            "Min Samples Leaf",
        ]
        self.parameters = [
            GenericVar(value=100),
            GenericVar(value=100),
            GenericVar(value=2),
            GenericVar(value=1),
        ]
        self.optimization_parameters = [
            [GenericVar(value=75), GenericVar(value=150)],
            [GenericVar(value=5), GenericVar(value=15)],
            [GenericVar(value=2), GenericVar(value=4)],
            [GenericVar(value=1), GenericVar(value=4)],
        ]

        ttk.Label(model_parameters_frame, text="Current").grid(column=1, row=0)
        ttk.Label(model_parameters_frame, text="----- Search Range -----").grid(
            column=2, row=0, columnspan=2
        )

        self.model_parameters_frame_options = [
            [
                ttk.Label(model_parameters_frame, text=j + ":").grid(
                    column=0, row=i + 1
                ),
                ttk.Entry(
                    model_parameters_frame,
                    textvariable=self.parameters[i],
                    state=tk.DISABLED,
                    width=9,
                ),
                ttk.Entry(
                    model_parameters_frame,
                    textvariable=self.optimization_parameters[i][0],
                    state=tk.DISABLED,
                    width=9,
                ),
                ttk.Entry(
                    model_parameters_frame,
                    textvariable=self.optimization_parameters[i][1],
                    state=tk.DISABLED,
                    width=9,
                ),
            ]
            for i, j in enumerate(parameter_names)
        ]

        for i, j in enumerate(self.model_parameters_frame_options):
            j[1].grid(column=1, row=i + 1, padx=2, pady=2, sticky=tk.W)
            j[2].grid(column=2, row=i + 1, padx=2, pady=2)
            j[3].grid(column=3, row=i + 1, padx=2, pady=2)

        ttk.Button(model_frame, text="Create Model", command=self.create_model).grid(
            column=0, row=3
        )
        ttk.Button(model_frame, text="Save Model", command=self.save_model).grid(
            column=1, row=3
        )
        ttk.Button(model_frame, text="Load Model", command=self.load_model).grid(
            column=2, row=3
        )

        # Test Model
        test_model_frame = ttk.LabelFrame(self.root, text="Test Frame")
        test_model_frame.grid(column=1, row=1)

        ## Test Model Main
        test_model_main_frame = ttk.LabelFrame(test_model_frame, text="Test Model")
        test_model_main_frame.grid(column=0, row=0)

        self.forecast_num = GenericVar(value="")  # type: ignore
        ttk.Label(test_model_main_frame, text="# of Forecast").grid(column=0, row=0)
        ttk.Entry(test_model_main_frame, textvariable=self.forecast_num).grid(
            column=1, row=0
        )
        ttk.Button(
            test_model_main_frame, text="Values", command=self.show_result_values
        ).grid(column=2, row=0)

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
            text="Actual vs Forecast Graph",
            command=self.show_result_graph,
        ).grid(column=0, row=4, columnspan=3)

        ## Test Model Metrics
        test_model_metrics_frame = ttk.LabelFrame(test_model_frame, text="Test Metrics")
        test_model_metrics_frame.grid(column=1, row=0)

        test_metrics: list[str] = ["NMSE", "RMSE", "MAE", "MAPE", "SMAPE"]
        self.test_metrics_vars = [
            GenericVar(value="") for _ in range(len(test_metrics))
        ]
        for i, j in enumerate(test_metrics):
            ttk.Label(test_model_metrics_frame, text=j).grid(column=0, row=i)
            ttk.Entry(
                test_model_metrics_frame, textvariable=self.test_metrics_vars[i]
            ).grid(column=1, row=i)

        self.__open_entries()
        self.__open_other_entries()

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

    def __open_entries(self):
        to_open = []
        for i in self.model_parameters_frame_options:
            i[1]["state"] = tk.DISABLED
            i[2]["state"] = tk.DISABLED
            i[3]["state"] = tk.DISABLED

        self.interval_entry["state"] = tk.DISABLED
        self.gs_cross_val_entry["state"] = tk.DISABLED

        if self.grid_option_var.get() and self.gs_cross_val_option.get():
            self.gs_cross_val_entry["state"] = tk.NORMAL

        to_open = list(range(4))
        opt = self.grid_option_var.get()
        self.__open(to_open, opt)

    def __open(self, to_open, opt=0):
        if opt == 1:
            self.interval_entry["state"] = tk.NORMAL
            for i in to_open:
                self.model_parameters_frame_options[i][2]["state"] = tk.NORMAL
                self.model_parameters_frame_options[i][3]["state"] = tk.NORMAL
        else:
            for i in to_open:
                self.model_parameters_frame_options[i][1]["state"] = tk.NORMAL

        self.vars_nums = to_open

    def __open_other_entries(self):
        if not self.do_forecast_option.get():
            self.cv_entry_1["state"] = tk.NORMAL
            self.cv_entry_2["state"] = tk.NORMAL
        else:
            self.cv_entry_1["state"] = tk.DISABLED
            self.cv_entry_2["state"] = tk.DISABLED
        if self.validation_option.get() == 1:
            self.random_percent_entry["state"] = tk.NORMAL
        else:
            self.random_percent_entry["state"] = tk.DISABLED
        if self.validation_option.get() == 2:
            self.cv_value_entry["state"] = tk.NORMAL
        else:
            self.cv_value_entry["state"] = tk.DISABLED
        if self.lookback_option.get():
            self.lookback_entry["state"] = tk.NORMAL
        else:
            self.lookback_entry["state"] = tk.DISABLED
            self.lookback_val_var.reset()
        if self.seasonal_lookback_option.get():
            self.seasonal_lookback_entry_1["state"] = tk.NORMAL
            self.seasonal_lookback_entry_2["state"] = tk.NORMAL
        else:
            self.seasonal_lookback_entry_1["state"] = tk.DISABLED
            self.seasonal_lookback_entry_2["state"] = tk.DISABLED
            self.seasonal_val_var.reset()
            self.seasonal_period_var.reset()

    def __check_errors(self):
        self.input_component.check_errors()

        try:
            msg = "Enter a valid percent value"
            if self.random_percent_var.get() <= 0:
                raise Exception

            msg = "Enter a valid K-fold value (Above 2)"
            if self.validation_option.get() == 2 and self.cross_val_var.get() <= 1:
                raise Exception

            msg = "Enter a valid lookback value"
            if self.lookback_option.get():
                self.lookback_val_var.get()

            msg = "Enter valid periodic lookback values"
            if self.seasonal_lookback_option.get():
                self.seasonal_val_var.get()
                self.seasonal_period_var.get()

            msg = "Enter a valid Interval for grid search"
            if self.grid_option_var.get() and self.interval_var.get() < 1:
                raise Exception

            msg = "Enter a valid Cross Validation fold for grid search (Above 2)"
            if self.gs_cross_val_option.get() and self.gs_cross_val_var.get() < 2:
                raise Exception

        except Exception:
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
        params["predictor_names"] = self.predictor_names
        params["label_name"] = self.label_name
        params["do_forecast"] = self.do_forecast_option.get()
        params["validation_option"] = self.validation_option.get()
        params["random_percent"] = (
            self.random_percent_var.get() if self.validation_option.get() == 1 else None
        )
        params["k_fold_cv"] = (
            self.cross_val_var.get() if self.validation_option.get() == 2 else None
        )
        params["lookback_option"] = self.lookback_option.get()
        params["lookback_value"] = (
            self.lookback_val_var.get() if self.lookback_option.get() else None
        )
        params["seasonal_lookback_option"] = self.seasonal_lookback_option.get()
        params["seasonal_period"] = (
            self.seasonal_period_var.get()
            if self.seasonal_lookback_option.get()
            else None
        )
        params["seasonal_value"] = (
            self.seasonal_val_var.get() if self.seasonal_lookback_option.get() else None
        )
        params["sliding"] = self.sliding

        params["scaler"] = self.data_scaler.get_params()

        os.mkdir(path)
        dump(self.model, path + "/model.joblib")
        if self.lookback_option.get() == 1:
            with open(path + "/last_values.npy", "wb") as outfile:
                np.save(outfile, self.last)
        if self.seasonal_lookback_option.get() == 1:
            with open(path + "/seasonal_last_values.npy", "wb") as outfile:
                np.save(outfile, self.seasonal_last)
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

        self.predictor_names = params.get("predictor_names", [])
        self.label_name = params.get("label_name", "")

        self.do_forecast_option.set(params.get("do_forecast", 1))
        self.validation_option.set(params.get("validation_option", 0))
        if self.validation_option.get() == 1:
            self.random_percent_var.set(params.get("random_percent", 80))
        elif self.validation_option.get() == 2:
            self.cross_val_var.set(params.get("k_fold_cv", 5))
        self.lookback_option.set(params.get("lookback_option", 0))
        if self.lookback_option.get() == 1:
            self.lookback_val_var.set(params.get("lookback_value", 0))
            try:
                with open(path + "/last_values.npy", "rb") as last_values:
                    self.last = np.load(last_values)
            except Exception:
                pass

        self.sliding = params.get("sliding", -1)
        self.seasonal_lookback_option.set(params.get("seasonal_lookback_option", 0))
        if self.seasonal_lookback_option.get() == 1:
            self.seasonal_period_var.set(params.get("seasonal_period", 0))
            self.seasonal_val_var.set(params.get("seasonal_value", 0))
            try:
                with open(path + "/seasonal_last_values.npy", "rb") as slv:
                    self.seasonal_last = np.load(slv)
            except Exception:
                pass

        self.scale_var.set(params.get("scaler", {"type": "None"}).get("type"))
        self.data_scaler.set_params(params.get("scaler", {"params": {}}).get("params"))

        self.parameters[0].set(params.get("n_estimators", 100))
        self.parameters[1].set(params.get("max_depth", 5))
        self.parameters[2].set(params.get("min_samples_split", 2))
        self.parameters[3].set(params.get("min_samples_leaf", 1))

        self.__open_entries()
        self.__open_other_entries()
        names = "\n".join(self.predictor_names)
        msg = f"Predictor names are {names}\nLabel name is {self.label_name}"
        popupmsg(msg)

    def __get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        self.data_scaler.initialize(self.scale_var.get())
        self.data_handler.set_names(
            self.input_component.get_predictors(), self.input_component.get_target()
        )

        self.lookback_handler.initialize(
            self.lookback_val_var.get(),
            self.seasonal_val_var.get(),
            self.seasonal_period_var.get(),
        )

        X, y = self.data_handler.get_Xy()
        X, y = self.data_scaler.scale(X, y)
        X, y = self.lookback_handler.get_lookback(X, y)

        return X, y

    def create_model(self):
        if self.__check_errors():
            return

        do_forecast = self.do_forecast_option.get()
        val_option = self.validation_option.get()

        X, y = self.__get_data()

        if self.grid_option_var.get() == 0:
            n_estimators = self.parameters[0].get()
            max_depth = self.parameters[1].get()
            min_samples_split = self.parameters[2].get()
            min_samples_leaf = self.parameters[3].get()

            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
            )

            if val_option == 0:
                model.fit(X, y)
                if do_forecast == 0:
                    pred = model.predict(X).reshape(-1)
                    if self.scale_var.get() != "None":
                        pred = self.data_scaler.inverse_scale(pred)
                        y = self.data_scaler.inverse_scale(y)
                    losses = loss(y, pred)
                    self.y_test = y
                    self.pred = pred
                    for i, j in enumerate(losses):
                        self.test_metrics_vars[i].set(j)
                self.model = model  # type: ignore

            elif val_option == 1:
                if do_forecast == 0:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, train_size=self.random_percent_var.get() / 100
                    )
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test).reshape(-1)
                    if self.scale_var.get() != "None":
                        pred = self.data_scaler.inverse_scale(pred)
                        y = self.data_scaler.inverse_scale(y)
                    losses = loss(y_test, pred)
                    self.y_test = y_test
                    self.pred = pred
                    for i, j in enumerate(losses):
                        self.test_metrics_vars[i].set(j)
                else:
                    size = int((self.random_percent_var.get() / 100) * len(X))
                    X = X[-size:]
                    y = y[-size:]
                    model.fit(X, y)
                self.model = model  # type: ignore

            elif val_option == 2:
                if do_forecast == 0:
                    cvs = cross_validate(
                        model, X, y, cv=self.cross_val_var.get(), scoring=skloss
                    )
                    for i, j in enumerate(list(cvs.values())[2:]):
                        self.test_metrics_vars[i].set(j.mean())

            elif val_option == 3:
                if do_forecast == 0:
                    cvs = cross_validate(model, X, y, cv=X.shape[0] - 1, scoring=skloss)
                    for i, j in enumerate(list(cvs.values())[2:]):
                        self.test_metrics_vars[i].set(j.mean())

        else:
            params = {}
            interval = self.interval_var.get()

            params["n_estimators"] = np.unique(
                np.linspace(
                    self.optimization_parameters[0][0].get(),
                    self.optimization_parameters[0][1].get(),
                    interval,
                    dtype=int,
                )
            )
            params["max_depth"] = np.unique(
                np.linspace(
                    self.optimization_parameters[1][0].get(),
                    self.optimization_parameters[1][1].get(),
                    interval,
                    dtype=int,
                )
            )
            params["min_samples_split"] = np.unique(
                np.linspace(
                    self.optimization_parameters[2][0].get(),
                    self.optimization_parameters[2][1].get(),
                    interval,
                    dtype=int,
                )
            )
            params["min_samples_leaf"] = np.unique(
                np.linspace(
                    self.optimization_parameters[3][0].get(),
                    self.optimization_parameters[3][1].get(),
                    interval,
                    dtype=int,
                )
            )

            cv = (
                self.gs_cross_val_var.get()
                if self.gs_cross_val_option.get() == 1
                else None
            )
            regressor = GridSearchCV(RandomForestRegressor(), params, cv=cv)

            if val_option == 0:
                regressor.fit(X, y)
                if do_forecast == 0:
                    pred = regressor.predict(X)
                    if self.scale_var.get() != "None":
                        pred = self.data_scaler.inverse_scale(pred)
                        y = self.data_scaler.inverse_scale(y)
                    losses = loss(y, pred)
                    self.y_test = y
                    self.pred = pred
                    for i, j in enumerate(losses):
                        self.test_metrics_vars[i].set(j)
                self.model = regressor.best_estimator_

            elif val_option == 1:
                if do_forecast == 0:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, train_size=self.random_percent_var.get() / 100
                    )
                    regressor.fit(X_train, y_train)
                    pred = regressor.predict(X_test)
                    if self.scale_var.get() != "None":
                        pred = self.data_scaler.inverse_scale(pred)
                        y = self.data_scaler.inverse_scale(y)
                    losses = loss(y_test, pred)
                    self.y_test = y_test
                    self.pred = pred
                    for i, j in enumerate(losses):
                        self.test_metrics_vars[i].set(j)
                else:
                    size = int((self.random_percent_var.get() / 100) * len(X))
                    X = X[-size:]
                    y = y[-size:]
                    regressor.fit(X, y)
                self.model = regressor.best_estimator_

            popupmsg("Best Params: " + str(self.model.get_params()))

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
            num = self.forecast_num.get()
        except Exception:
            popupmsg("Enter a valid forecast value")
            return

        try:
            X_test, y_test = self.data_handler.get_test_Xy(num)
            self.y_test = y_test
        except Exception:
            popupmsg("Read a test data")
            return

        X_test, _ = self.data_scaler.scale(X_test, np.ones(num))
        self.pred = self.__predict(X_test, num)
        self.pred = self.data_scaler.inverse_scale(self.pred)

        losses = loss(y_test, self.pred)
        for i in range(len(self.test_metrics_vars)):
            self.test_metrics_vars[i].set(losses[i])

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
