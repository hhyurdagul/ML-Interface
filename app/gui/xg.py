import json
import os
import tkinter as tk
from tkinter import ttk, filedialog

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd
from joblib import dump, load  # type: ignore
from pandastable import Table  # type: ignore

from .backend import (
    DataHandler,
    ScalerHandler,
    LookbackHandler,
    ModelHandler,
    handle_errors,
)
from .components import (
    InputListComponent,
    ModelValidationComponent,
    CustomizeTrainSetComponent,
    ModelComponent,
)
from .helpers import loss, popupmsg


class XGB:
    def __init__(self):
        self.root = ttk.Frame()
        self.data_handler = DataHandler()
        self.scaler_handler = ScalerHandler()
        self.lookback_handler = LookbackHandler()
        self.model_handler = ModelHandler()

        self.input_list_component = InputListComponent(self.root, self.data_handler)
        self.model_validation_component = ModelValidationComponent(self.root)
        self.customize_train_set_component = CustomizeTrainSetComponent(
            self.root, self.scaler_handler, self.lookback_handler
        )
        self.model_component = ModelComponent(self.root, self.model_handler)

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
        self.data_handler.read_test_data(path)

    def save_model(self):
        path = filedialog.asksaveasfilename()
        if not path:
            return
        try:
            if self.model_component.grid_option_var.get():
                print(self.best_params)
                model_params = {
                    "n_estimators": self.best_params["n_estimators"],
                    "max_depth": self.best_params["max_depth"],
                    "learning_rate": self.best_params["learning_rate"],
                }
            else:
                model_params = {
                    "n_estimators": self.parameters["n_estimators"].get("var").get(),
                    "max_depth": self.parameters["max_depth"].get("var").get(),
                    "learning_rate": self.parameters["learning_rate"].get("var").get(),
                }

        except Exception:
            popupmsg("Model is not created")
            return

        save_params = {}
        save_params.update(model_params)
        save_params.update(self.input_list_component.get_params())
        save_params.update(self.model_validation_component.get_params())
        save_params.update(self.customize_train_set_component.get_params())

        save_params["is_round"] = self.is_round
        save_params["is_negative"] = self.is_negative

        os.mkdir(path)
        self.customize_train_set_component.save_files(path)

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
        names = "\n".join(self.predictor_names)
        msg = f"Predictor names are {names}\nLabel name is {self.label_name}"
        popupmsg(msg)

    def __check_errors(self):
        return handle_errors(
            self.input_list_component.check_errors,
            self.model_validation_component.check_errors,
            self.customize_train_set_component.check_errors,
            self.model_component.check_errors,
        )

    def __set_handlers(self):
        self.data_handler.set_variables(
            self.input_list_component.get_predictor_names(),
            self.input_list_component.get_target_name(),
            self.model_validation_component.validation_option.get(),
            self.model_validation_component.do_forecast_option.get(),
            self.model_validation_component.random_percent_var.get(),
            self.model_validation_component.cross_val_var.get(),
        )

        self.scaler_handler.set_scalers(
            self.customize_train_set_component.scale_var.get()
        )

        self.customize_train_set_component.calculate_sliding()
        self.lookback_handler.set_variables(
            self.customize_train_set_component.lookback_var.get(),
            self.customize_train_set_component.seasonal_period_var.get(),
            self.customize_train_set_component.seasonal_val_var.get(),
            self.customize_train_set_component.sliding,
        )

        self.model_handler.set_variables(
                self.model_component.get_model_params(),
                self.model_component.get_grid_params(),
                self.model_component.grid_option_var.get()
        )

    def create_model(self):
        if not self.__check_errors():
            return

        self.__set_handlers()
        self.model_handler.create_model()
        
        for i, j in enumerate(self.model_handler.loss):
            self.test_metrics_vars[i].set(j)

        if self.model_handler.grid_option:
            p = self.model_handler.best_params()
            self.best_params = {
                "n_estimators": int(p["n_estimators"]),
                "max_depth": int(p["max_depth"]),
                "learning_rate": float(p["learning_rate"]),
            }
            popupmsg("Best Params: " + str(self.best_params))

    def __forecast_lookback(
        self, num, lookback=0, seasons=0, seasonal_lookback=0, sliding=-1
    ):
        pred = []
        if sliding == 0:
            last = self.lookback_handler.last
            for i in range(num):
                X_test = self.data_handler.test_df[self.predictor_names].iloc[i]
                if self.customize_train_set_component.scale_var.get() != "None":
                    X_test.iloc[:] = self.scaler_handler.feature_scaler.transform(
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
            seasonal_last = self.lookback_handler.seasonal_last
            for i in range(num):
                X_test = self.data_handler.test_df[self.predictor_names].iloc[i]
                if self.customize_train_set_component.scale_var.get() != "None":
                    X_test.iloc[:] = self.scaler_handler.feature_scaler.transform(
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
            last = self.lookback_handler.last
            seasonal_last = self.lookback_handler.seasonal_last
            for i in range(num):
                X_test = self.data_handler.test_df[self.predictor_names].iloc[i]
                if self.customize_train_set_component.scale_var.get() != "None":
                    X_test.iloc[:] = self.scaler_handler.feature_scaler.transform(
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
            X_test = self.data_handler.test_df[self.predictor_names][:num].to_numpy()
            y_test = (
                self.data_handler.test_df[self.label_name][:num].to_numpy().reshape(-1)
            )
            self.y_test = y_test
        except Exception:
            popupmsg("Read a test data")
            return

        if self.customize_train_set_component.sliding == -1:
            if self.customize_train_set_component.scale_var.get() != "None":
                X_test = self.scaler_handler.feature_scaler.transform(X_test)

            self.pred = self.model.predict(X_test).reshape(-1)
        else:
            sliding = self.customize_train_set_component.sliding
            lookback = self.customize_train_set_component.lookback_val_var.get()
            seasonal_lookback = (
                self.customize_train_set_component.seasonal_val_var.get()
            )
            seasons = self.customize_train_set_component.seasonal_period_var.get()

            self.pred = self.__forecast_lookback(
                num, lookback, seasons, seasonal_lookback, sliding
            )

        if self.customize_train_set_component.scale_var.get() != "None":
            self.pred = self.scaler_handler.label_inverse_transform(self.pred)

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
