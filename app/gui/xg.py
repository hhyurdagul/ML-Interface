import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from pandastable import Table  # type: ignore

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import dump, load  # type: ignore
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # type: ignore
from sklearn.model_selection import GridSearchCV, train_test_split, cross_validate

import os
import json
from pickle import dump as pickle_dump
from pickle import load as pickle_load

from .helpers import loss, skloss, popupmsg
from .backend import DataHandler, handle_errors
from .components import InputListComponent, ModelValidationComponent, CustomizeTrainSetComponent



class XGB:
    def __init__(self):
        self.root = ttk.Frame()
        self.data_handler = DataHandler()

        self.input_list_component = InputListComponent(self.root, self.data_handler)
        self.model_validation_component = ModelValidationComponent(self.root)
        self.customize_train_set_component = CustomizeTrainSetComponent(self.root)

        # Model
        model_frame = ttk.Labelframe(self.root, text="Model Frame")
        model_frame.grid(column=1, row=0)

        ## Parameter Optimization
        parameter_optimization_frame = ttk.Labelframe(
            model_frame, text="Parameter Optimization"
        )
        parameter_optimization_frame.grid(column=0, row=2)

        self.grid_option_var = tk.IntVar(value=0)
        tk.Checkbutton(
            parameter_optimization_frame,
            text="Do grid search for optimal parameters",
            offvalue=0,
            onvalue=1,
            variable=self.grid_option_var,
            command=self.__open_entries,
        ).grid(column=0, row=0, columnspan=3)

        self.interval_var = tk.IntVar(value=3)
        ttk.Label(parameter_optimization_frame, text="Interval:").grid(column=0, row=1)
        self.interval_entry = ttk.Entry(
            parameter_optimization_frame,
            textvariable=self.interval_var,
            width=8,
            state=tk.DISABLED,
        )
        self.interval_entry.grid(column=1, row=1, pady=2)

        self.gs_cross_val_option = tk.IntVar(value=0)
        self.gs_cross_val_var = tk.IntVar(value=5)
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

        parameter_names = ["N Estimators", "Max Depth", "Learning Rate"]
        self.parameters = [
            tk.IntVar(value=100),
            tk.IntVar(value=6),
            tk.DoubleVar(value=0.3),
        ]
        self.optimization_parameters = [
            [tk.IntVar(value=75), tk.IntVar(value=150)],
            [tk.IntVar(value=5), tk.IntVar(value=10)],
            [tk.DoubleVar(value=0.2), tk.DoubleVar(value=1)],
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

        self.forecast_num = tk.IntVar(value="")  # type: ignore
        ttk.Label(test_model_main_frame, text="# of Forecast").grid(column=0, row=0)
        ttk.Entry(test_model_main_frame, textvariable=self.forecast_num).grid(
            column=1, row=0
        )
        ttk.Button(
            test_model_main_frame, text="Values", command=self.show_result_values
        ).grid(column=2, row=0)

        test_file_path = tk.StringVar()
        ttk.Label(test_model_main_frame, text="Test File Path").grid(column=0, row=1)
        ttk.Entry(test_model_main_frame, textvariable=test_file_path).grid(
            column=1, row=1
        )
        ttk.Button(
            test_model_main_frame,
            text="Get Test Set",
            command=lambda: self.get_test_data(test_file_path),
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

        test_metrics = ["NMSE", "RMSE", "MAE", "MAPE", "SMAPE"]
        self.test_metrics_vars = [tk.Variable() for _ in range(len(test_metrics))]
        for i, j in enumerate(test_metrics):
            ttk.Label(test_model_metrics_frame, text=j).grid(column=0, row=i)
            ttk.Entry(
                test_model_metrics_frame, textvariable=self.test_metrics_vars[i]
            ).grid(column=1, row=i)

        self.__open_entries()

    def get_test_data(self, file_path):
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
        if path.endswith(".csv"):
            self.test_df = pd.read_csv(path)  # type: ignore
        else:
            try:
                self.test_df = pd.read_excel(path)
            except Exception:
                self.test_df = pd.read_excel(path, engine="openpyxl")

    def save_model(self):
        path = filedialog.asksaveasfilename()
        if not path:
            return
        try:
            if self.grid_option_var.get():
                print(self.best_params)
                model_params = {
                    "n_estimators": self.best_params["n_estimators"],
                    "max_depth": self.best_params["max_depth"],
                    "learning_rate": self.best_params["learning_rate"],
                }
            else:
                model_params = {
                    "n_estimators": self.parameters[0].get(),
                    "max_depth": self.parameters[1].get(),
                    "learning_rate": self.parameters[2].get(),
                }

        except Exception:
            popupmsg("Model is not created")
            return
        
        files = {
            "last": self.last,
            "seasonal_last": self.seasonal_last,
            "feature_scaler": self.feature_scaler,
            "label_scaler": self.label_scaler
        }

        save_params = {}
        save_params.update(model_params)
        save_params.update(self.input_list_component.get_save_dict())
        save_params.update(self.model_validation_component.get_params())
        save_params.update(self.customize_train_set_component.get_params())

        save_params["is_round"] = self.is_round
        save_params["is_negative"] = self.is_negative

        self.customize_train_set_component.save_files(path, files)

        os.mkdir(path)
        dump(self.model, path + "/model.joblib")
        with open(path + "/model.json", "w") as outfile:
            json.dump(save_params, outfile)

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

        self.is_round = params.get("is_round", True)
        self.is_negative = params.get("is_negative", False)

        self.model_validation_component.set_params(params)
        self.customize_train_set_component.set_params(params)

        self.customize_train_set_component.load_files(path)

        self.parameters[0].set(params.get("n_estimators", 100))
        self.parameters[1].set(params.get("max_depth", 5))
        self.parameters[2].set(params.get("learning_rate", 0.3))

        self.__open_entries()
        self.__open_other_entries()
        names = "\n".join(self.predictor_names)
        msg = f"Predictor names are {names}\nLabel name is {self.label_name}"
        popupmsg(msg)

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

        to_open = list(range(len(self.parameters)))
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

    def __check_errors(self):
        if handle_errors(
            self.input_list_component.check_errors,
            self.model_validation_component.check_errors,
            self.customize_train_set_component.check_errors,
        ):
            return True
        else:
            try:
                msg = "Enter a valid Interval for grid search"
                if self.grid_option_var.get() and self.interval_var.get() < 1:
                    raise Exception

                msg = "Enter a valid Cross Validation fold for grid search (Above 2)"
                if self.gs_cross_val_option.get() and self.gs_cross_val_var.get() < 2:
                    raise Exception

            except Exception:
                popupmsg(msg)  # type: ignore
                return False

    def __get_lookback(
        self, X, y, lookback=0, seasons=0, seasonal_lookback=0, sliding=-1
    ):
        if sliding == 0:
            for i in range(1, lookback + 1):
                X[f"t-{i}"] = y.shift(i)
        elif sliding == 1:
            for i in range(1, seasons + 1):
                X[f"t-{i*seasonal_lookback}"] = y.shift(i * seasonal_lookback)
        elif sliding == 2:
            for i in range(1, lookback + 1):
                X[f"t-{i}"] = y.shift(i)
            for i in range(1, seasons + 1):
                X[f"t-{i*seasonal_lookback}"] = y.shift(i * seasonal_lookback)

        X.dropna(inplace=True)
        a = X.to_numpy()
        b = y.iloc[-len(a) :].to_numpy().reshape(-1)

        if sliding == 0:
            self.last = b[-lookback:]
        elif sliding == 1:
            self.seasonal_last = b[-seasonal_lookback * seasons :]
        elif sliding == 2:
            self.last = b[-(lookback + seasonal_lookback) : -seasonal_lookback]
            self.seasonal_last = b[-seasonal_lookback * seasons :]

        return a, b

    def __get_data(self):
        self.is_round = False
        self.is_negative = False
        lookback_option = self.customize_train_set_component.lookback_option.get()
        seasonal_lookback_option = self.customize_train_set_component.seasonal_lookback_option.get()
        sliding = lookback_option + 2 * seasonal_lookback_option - 1
        scale_choice = self.customize_train_set_component.scale_var.get()

        self.predictor_names = self.input_list_component.get_predictor_names()
        self.label_name = self.input_list_component.get_target_name()

        df = self.data_handler.df

        X = df[self.predictor_names].copy()
        y = df[self.label_name].copy()

        if y.dtype == int or y.dtype == np.intc or y.dtype == np.int64:
            self.is_round = True
        if any(y < 0):
            self.is_negative = True

        if scale_choice == "StandardScaler":
            self.feature_scaler = StandardScaler()
            self.label_scaler = StandardScaler()

            X.iloc[:] = self.feature_scaler.fit_transform(X)
            y.iloc[:] = self.label_scaler.fit_transform(
                y.values.reshape(-1, 1)
            ).reshape(-1)

        elif scale_choice == "MinMaxScaler":
            self.feature_scaler = MinMaxScaler()
            self.label_scaler = MinMaxScaler()

            X.iloc[:] = self.feature_scaler.fit_transform(X)
            y.iloc[:] = self.label_scaler.fit_transform(
                y.values.reshape(-1, 1)
            ).reshape(-1)

        lookback = self.customize_train_set_component.lookback_val_var.get()
        seasonal_period = self.customize_train_set_component.seasonal_period_var.get()
        seasonal_lookback = self.customize_train_set_component.seasonal_val_var.get()

        X, y = self.__get_lookback(
            X, y, lookback, seasonal_period, seasonal_lookback, sliding
        )

        return X, y

    def create_model(self):
        if not self.__check_errors():
            return

        do_forecast = self.model_validation_component.do_forecast_option.get()
        val_option = self.model_validation_component.validation_option.get()

        X, y = self.__get_data()
        X: np.ndarray
        y: np.ndarray

        if self.grid_option_var.get() == 0:
            n_estimators = self.parameters[0].get()
            max_depth = self.parameters[1].get()
            learning_rate = self.parameters[2].get()

            model = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
            )

            if val_option == 0:
                model.fit(X, y)
                if do_forecast == 0:
                    pred = model.predict(X).reshape(-1)
                    if self.customize_train_set_component.scale_var.get() != "None":
                        pred = self.label_scaler.inverse_transform(
                            pred.reshape(-1, 1)
                        ).reshape(
                            -1
                        )  # type: ignore
                        y = self.label_scaler.inverse_transform(
                            y.reshape(-1, 1)
                        ).reshape(
                            -1
                        )  # type: ignore
                    losses = loss(y, pred)
                    self.y_test = y
                    self.pred = pred
                    for i, j in enumerate(losses):
                        self.test_metrics_vars[i].set(j)
                self.model = model  # type: ignore

            elif val_option == 1:
                if do_forecast == 0:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y,
                        train_size=self.model_validation_component.random_percent_var.get()
                        / 100,
                    )
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test).reshape(-1)
                    if self.customize_train_set_component.scale_var.get() != "None":
                        pred = self.label_scaler.inverse_transform(
                            pred.reshape(-1, 1)
                        ).reshape(
                            -1
                        )  # type: ignore
                        y_test = self.label_scaler.inverse_transform(
                            y_test.reshape(-1, 1)
                        ).reshape(
                            -1
                        )  # type: ignore
                    losses = loss(y_test, pred)
                    self.y_test = y_test
                    self.pred = pred
                    for i, j in enumerate(losses):
                        self.test_metrics_vars[i].set(j)
                else:
                    size = int(
                        (self.model_validation_component.random_percent_var.get() / 100)
                        * len(X)
                    )
                    X = X[-size:]
                    y = y[-size:]
                    model.fit(X, y)
                self.model = model  # type: ignore

            elif val_option == 2:
                if do_forecast == 0:
                    cvs = cross_validate(
                        model,
                        X,
                        y,
                        cv=self.model_validation_component.cross_val_var.get(),
                        scoring=skloss,
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
            params["learning_rate"] = np.unique(
                np.linspace(
                    self.optimization_parameters[2][0].get(),
                    self.optimization_parameters[2][1].get(),
                    interval,
                    dtype=float,
                )
            )

            cv = (
                self.gs_cross_val_var.get()
                if self.gs_cross_val_option.get() == 1
                else None
            )
            regressor = GridSearchCV(XGBRegressor(), params, cv=cv)

            if val_option == 0:
                regressor.fit(X, y)
                if do_forecast == 0:
                    pred = regressor.predict(X)
                    if self.customize_train_set_component.scale_var.get() != "None":
                        pred = self.label_scaler.inverse_transform(
                            pred.reshape(-1, 1)
                        ).reshape(
                            -1
                        )  # type: ignore
                        y = self.label_scaler.inverse_transform(
                            y.reshape(-1, 1)
                        ).reshape(
                            -1
                        )  # type: ignore
                    losses = loss(y, pred)
                    self.y_test = y
                    self.pred = pred
                    for i, j in enumerate(losses):
                        self.test_metrics_vars[i].set(j)
                self.model = regressor.best_estimator_

            elif val_option == 1:
                if do_forecast == 0:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y,
                        train_size=self.model_validation_component.random_percent_var.get()
                        / 100,
                    )
                    regressor.fit(X_train, y_train)
                    pred = regressor.predict(X_test)
                    if self.customize_train_set_component.scale_var.get() != "None":
                        pred = self.label_scaler.inverse_transform(
                            pred.reshape(-1, 1)
                        ).reshape(
                            -1
                        )  # type: ignore
                        y_test = self.label_scaler.inverse_transform(
                            y_test.reshape(-1, 1)
                        ).reshape(
                            -1
                        )  # type: ignore
                    losses = loss(y_test, pred)
                    self.y_test = y_test
                    self.pred = pred
                    for i, j in enumerate(losses):
                        self.test_metrics_vars[i].set(j)
                else:
                    size = int(
                        (self.model_validation_component.random_percent_var.get() / 100)
                        * len(X)
                    )
                    X = X[-size:]
                    y = y[-size:]
                    regressor.fit(X, y)
                self.model = regressor.best_estimator_

            p = self.model.get_params()
            self.best_params = {
                "n_estimators": int(p["n_estimators"]),
                "max_depth": int(p["max_depth"]),
                "learning_rate": float(p["learning_rate"]),
            }
            popupmsg("Best Params: " + str(self.best_params))

    def __forecast_lookback(
        self, num, lookback=0, seasons=0, seasonal_lookback=0, sliding=-1
    ):
        self.test_df: pd.DataFrame
        pred = []
        if sliding == 0:
            last = self.last
            for i in range(num):
                X_test = self.test_df[self.predictor_names].iloc[i]
                if self.customize_train_set_component.scale_var.get() != "None":
                    X_test.iloc[:] = self.feature_scaler.transform(
                        X_test.values.reshape(1, -1)
                    ).reshape(
                        -1
                    )  # type: ignore
                for j in range(1, lookback + 1):
                    X_test[f"t-{j}"] = last[-j]  # type: ignore
                to_pred = X_test.to_numpy().reshape(1, -1)  # type: ignore
                out = self.model.predict(to_pred)
                last = np.append(last, out)[-lookback:]
                pred.append(out)

        elif sliding == 1:
            seasonal_last = self.seasonal_last
            for i in range(num):
                X_test = self.test_df[self.predictor_names].iloc[i]
                if self.customize_train_set_component.scale_var.get() != "None":
                    X_test.iloc[:] = self.feature_scaler.transform(
                        X_test.values.reshape(1, -1)
                    ).reshape(
                        -1
                    )  # type: ignore
                for j in range(1, seasons + 1):
                    X_test[f"t-{j*seasonal_last}"] = seasonal_last[
                        -j * seasonal_lookback
                    ]  # type: ignore
                to_pred = X_test.to_numpy().reshape(1, -1)  # type: ignore
                out = self.model.predict(to_pred)
                seasonal_last = np.append(seasonal_last, out)[1:]
                pred.append(out)

        elif sliding == 2:
            last = self.last
            seasonal_last = self.seasonal_last
            for i in range(num):
                X_test = self.test_df[self.predictor_names].iloc[i]
                if self.customize_train_set_component.scale_var.get() != "None":
                    X_test.iloc[:] = self.feature_scaler.transform(
                        X_test.values.reshape(1, -1)
                    ).reshape(
                        -1
                    )  # type: ignore
                for j in range(1, lookback + 1):
                    X_test[f"t-{j}"] = last[-j]  # type: ignore
                for j in range(1, seasons + 1):
                    X_test[f"t-{j*seasonal_lookback}"] = seasonal_last[
                        -j * seasonal_lookback
                    ]  # type: ignore
                to_pred = X_test.to_numpy().reshape(1, -1)  # type: ignore
                out = self.model.predict(to_pred)
                last = np.append(last, out)[-lookback:]
                seasonal_last = np.append(seasonal_last, out)[1:]
                pred.append(out)

        return np.array(pred).reshape(-1)

    def forecast(self):
        try:
            num = self.forecast_num.get()
        except Exception:
            popupmsg("Enter a valid forecast value")
            return
        try:
            X_test = self.test_df[self.predictor_names][:num].to_numpy()  # type: ignore
            y_test = self.test_df[self.label_name][:num].to_numpy().reshape(-1)
            self.y_test = y_test
        except Exception:
            popupmsg("Read a test data")
            return

        if (
            self.customize_train_set_component.lookback_option == 0
            and self.customize_train_set_component.seasonal_lookback_option == 0
        ):
            if self.customize_train_set_component.scale_var.get() != "None":
                X_test = self.feature_scaler.transform(X_test)
            self.pred = self.model.predict(X_test).reshape(-1)
        else:
            sliding = self.customize_train_set_component.sliding
            lookback = self.customize_train_set_component.lookback_val_var.get()
            seasonal_lookback = self.customize_train_set_component.seasonal_val_var.get()
            seasons = self.customize_train_set_component.seasonal_period_var.get()

            self.pred = self.__forecast_lookback(
                num, lookback, seasons, seasonal_lookback, sliding
            )

        if self.customize_train_set_component.scale_var.get() != "None":
            self.pred = self.label_scaler.inverse_transform(
                self.pred.reshape(-1, 1)
            ).reshape(
                -1
            )  # type: ignore

        if not self.is_negative:
            self.pred = self.pred.clip(0, None)
        if self.is_round:
            self.pred = np.round(self.pred).astype(int)

        losses = loss(y_test, self.pred)
        for i in range(len(self.test_metrics_vars)):
            self.test_metrics_vars[i].set(losses[i])

    def show_result_values(self):
        try:
            df = pd.DataFrame({"Test": self.y_test, "Predict": self.pred})
        except Exception:
            return
        top = tk.Toplevel(self.root)
        pt = Table(top, dataframe=df, editable=False)
        pt.show()

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
