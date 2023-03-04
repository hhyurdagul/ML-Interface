import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from pandastable import Table

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import dump, load
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split, cross_validate

import os
import json
from pickle import dump as pickle_dump
from pickle import load as pickle_load

from .helpers import *


class XGB:
    def __init__(self):
        self.root = ttk.Frame()

        # Get Train Set
        get_train_set_frame = ttk.Labelframe(self.root, text="Get Train Set")
        get_train_set_frame.grid(column=0, row=0)

        file_path = tk.StringVar(value="")
        ttk.Label(get_train_set_frame, text="Train File Path").grid(column=0, row=0)
        ttk.Entry(get_train_set_frame, textvariable=file_path).grid(column=1, row=0)
        ttk.Button(
            get_train_set_frame,
            text="Read Data",
            command=lambda: self.readCsv(file_path),
        ).grid(column=2, row=0)

        self.input_list = tk.Listbox(get_train_set_frame)
        self.input_list.grid(column=0, row=1)
        self.input_list.bind("<Double-Button-1>", self.addPredictor)
        self.input_list.bind("<Double-Button-3>", self.addTarget)

        self.predictor_list = tk.Listbox(get_train_set_frame)
        self.predictor_list.grid(column=1, row=1)
        self.predictor_list.bind("<Double-Button-1>", self.ejectPredictor)

        self.target_list = tk.Listbox(get_train_set_frame)
        self.target_list.grid(column=2, row=1)
        self.target_list.bind("<Double-Button-1>", self.ejectTarget)

        ttk.Button(
            get_train_set_frame, text="Add Predictor", command=self.addPredictor
        ).grid(column=1, row=2)
        ttk.Button(
            get_train_set_frame, text="Eject Predictor", command=self.ejectPredictor
        ).grid(column=1, row=3)

        ttk.Button(get_train_set_frame, text="Add Target", command=self.addTarget).grid(
            column=2, row=2
        )
        ttk.Button(
            get_train_set_frame, text="Eject Target", command=self.ejectTarget
        ).grid(column=2, row=3)

        # Model testing and validation
        model_validation_frame = ttk.Labelframe(
            self.root, text="Model testing and validation"
        )
        model_validation_frame.grid(column=0, row=1)

        self.do_forecast_option = tk.IntVar(value=0)
        tk.Checkbutton(
            model_validation_frame,
            text="Do Forecast",
            offvalue=0,
            onvalue=1,
            variable=self.do_forecast_option,
            command=self.openOtherEntries,
        ).grid(column=0, row=0, columnspan=2)

        self.validation_option = tk.IntVar(value=0)
        self.random_percent_var = tk.IntVar(value=70)
        self.cross_val_var = tk.IntVar(value=5)
        tk.Radiobutton(
            model_validation_frame,
            text="No validation, use all data rows",
            value=0,
            variable=self.validation_option,
            command=self.openOtherEntries,
        ).grid(column=0, row=1, columnspan=2, sticky=tk.W)
        tk.Radiobutton(
            model_validation_frame,
            text="Random percent",
            value=1,
            variable=self.validation_option,
            command=self.openOtherEntries,
        ).grid(column=0, row=2, sticky=tk.W)
        self.cv_entry_1 = tk.Radiobutton(
            model_validation_frame,
            text="K-fold cross-validation",
            value=2,
            variable=self.validation_option,
            command=self.openOtherEntries,
        )
        self.cv_entry_1.grid(column=0, row=3, sticky=tk.W)
        self.cv_entry_2 = tk.Radiobutton(
            model_validation_frame,
            text="Leave one out cross-validation",
            value=3,
            variable=self.validation_option,
            command=self.openOtherEntries,
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

        self.lookback_option = tk.IntVar(value=0)
        self.lookback_val_var = tk.IntVar(value="")  # type: ignore
        tk.Checkbutton(
            customize_train_set_frame,
            text="Lookback",
            offvalue=0,
            onvalue=1,
            variable=self.lookback_option,
            command=self.openOtherEntries,
        ).grid(column=0, row=0)
        self.lookback_entry = tk.Entry(
            customize_train_set_frame,
            textvariable=self.lookback_val_var,
            width=8,
            state=tk.DISABLED,
        )
        self.lookback_entry.grid(column=1, row=0)

        self.seasonal_lookback_option = tk.IntVar(value=0)
        self.seasonal_period_var = tk.IntVar(value="")  # type: ignore
        self.seasonal_val_var = tk.IntVar(value="")  # type: ignore
        tk.Checkbutton(
            customize_train_set_frame,
            text="Periodic Lookback",
            offvalue=0,
            onvalue=1,
            variable=self.seasonal_lookback_option,
            command=self.openOtherEntries,
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

        self.grid_option_var = tk.IntVar(value=0)
        tk.Checkbutton(
            parameter_optimization_frame,
            text="Do grid search for optimal parameters",
            offvalue=0,
            onvalue=1,
            variable=self.grid_option_var,
            command=self.openEntries,
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
            command=self.openEntries,
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

        ttk.Button(model_frame, text="Create Model", command=self.createModel).grid(
            column=0, row=3
        )
        ttk.Button(model_frame, text="Save Model", command=self.saveModel).grid(
            column=1, row=3
        )
        ttk.Button(model_frame, text="Load Model", command=self.loadModel).grid(
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
            test_model_main_frame, text="Values", command=self.showPredicts
        ).grid(column=2, row=0)

        test_file_path = tk.StringVar()
        ttk.Label(test_model_main_frame, text="Test File Path").grid(column=0, row=1)
        ttk.Entry(test_model_main_frame, textvariable=test_file_path).grid(
            column=1, row=1
        )
        ttk.Button(
            test_model_main_frame,
            text="Get Test Set",
            command=lambda: self.getTestSet(test_file_path),
        ).grid(column=2, row=1)

        ttk.Button(
            test_model_main_frame, text="Test Model", command=self.forecast
        ).grid(column=2, row=3)
        ttk.Button(
            test_model_main_frame, text="Actual vs Forecast Graph", command=self.vsGraph
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

        self.openEntries()
        self.openOtherEntries()

    def readCsv(self, file_path):
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
            self.df = pd.read_csv(path)  # type: ignore
        else:
            try:
                self.df = pd.read_excel(path)
            except Exception:
                self.df = pd.read_excel(path, engine="openpyxl")
        self.fillInputList()

    def fillInputList(self):
        self.input_list.delete(0, tk.END)

        self.df: pd.DataFrame
        for i in self.df.columns.to_list():
            self.input_list.insert(tk.END, i)

    def getTestSet(self, file_path):
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

    def showPredicts(self):
        try:
            df = pd.DataFrame({"Test": self.y_test, "Predict": self.pred})
        except Exception:
            return
        top = tk.Toplevel(self.root)
        pt = Table(top, dataframe=df, editable=False)
        pt.show()

    def addPredictor(self, _=None):
        try:
            a = self.input_list.get(self.input_list.curselection())
            if a not in self.predictor_list.get(0, tk.END):
                self.predictor_list.insert(tk.END, a)
        except Exception:
            pass

    def ejectPredictor(self, _=None):
        try:
            self.predictor_list.delete(self.predictor_list.curselection())
        except Exception:
            pass

    def addTarget(self, _=None):
        try:
            a = self.input_list.get(self.input_list.curselection())
            if self.target_list.size() < 1:
                self.target_list.insert(tk.END, a)
        except Exception:
            pass

    def ejectTarget(self, _=None):
        try:
            self.target_list.delete(self.target_list.curselection())
        except Exception:
            pass

    def saveModel(self):
        path = filedialog.asksaveasfilename()
        if not path:
            return
        try:
            if self.grid_option_var.get():
                print(self.best_params)
                params = {
                    "n_estimators": self.best_params["n_estimators"],
                    "max_depth": self.best_params["max_depth"],
                    "learning_rate": self.best_params["learning_rate"],
                }
            else:
                params = {
                    "n_estimators": self.parameters[0].get(),
                    "max_depth": self.parameters[1].get(),
                    "learning_rate": self.parameters[2].get(),
                }

        except Exception:
            popupmsg("Model is not created")
            return
        params["predictor_names"] = self.predictor_names
        params["label_name"] = self.label_name
        params["is_round"] = self.is_round
        params["is_negative"] = self.is_negative
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
        params["scale_type"] = self.scale_var.get()

        os.mkdir(path)
        dump(self.model, path + "/model.joblib")
        if self.scale_var.get() != "None":
            with open(path + "/feature_scaler.pkl", "wb") as f:
                pickle_dump(self.feature_scaler, f)
            with open(path + "/label_scaler.pkl", "wb") as f:
                pickle_dump(self.label_scaler, f)
        if self.lookback_option.get() == 1:
            with open(path + "/last_values.npy", "wb") as outfile:
                np.save(outfile, self.last)
        if self.seasonal_lookback_option.get() == 1:
            with open(path + "/seasonal_last_values.npy", "wb") as outfile:
                np.save(outfile, self.seasonal_last)
        with open(path + "/model.json", "w") as outfile:
            json.dump(params, outfile)

    def loadModel(self):
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

        self.predictor_names = params["predictor_names"]
        self.label_name = params["label_name"]
        try:
            self.is_round = params["is_round"]
        except Exception:
            self.is_round = True
        try:
            self.is_negative = params["is_negative"]
        except Exception:
            self.is_negative = False
        self.do_forecast_option.set(params["do_forecast"])
        self.validation_option.set(params["validation_option"])
        if params["validation_option"] == 1:
            self.random_percent_var.set(params["random_percent"])
        elif params["validation_option"] == 2:
            self.cross_val_var.set(params["k_fold_cv"])
        self.lookback_option.set(params["lookback_option"])
        self.sliding = -1
        if params["lookback_option"] == 1:
            self.lookback_val_var.set(params["lookback_value"])
            last_values = open(path + "/last_values.npy", "rb")
            self.last = np.load(last_values)
            last_values.close()
        try:
            self.sliding = params["sliding"]
            self.seasonal_lookback_option.set(params["seasonal_lookback_option"])
            if params["seasonal_lookback_option"] == 1:
                self.seasonal_period_var.set(params["seasonal_period"])
                self.seasonal_val_var.set(params["seasonal_value"])
                seasonal_last_values = open(path + "/seasonal_last_values.npy", "rb")
                self.seasonal_last = np.load(seasonal_last_values)
                seasonal_last_values.close()
        except Exception:
            pass
        self.scale_var.set(params["scale_type"])
        if params["scale_type"] != "None":
            try:
                with open(path + "/feature_scaler.pkl", "rb") as f:
                    self.feature_scaler = pickle_load(f)
                with open(path + "/label_scaler.pkl", "rb") as f:
                    self.label_scaler = pickle_load(f)
            except Exception:
                pass
        self.parameters[0].set(params["n_estimators"])
        self.parameters[1].set(params["max_depth"])
        self.parameters[2].set(params["learning_rate"])

        self.openEntries()
        self.openOtherEntries()
        msg = f"Predictor names are {self.predictor_names}\nLabel name is {self.label_name}"
        popupmsg(msg)

    def openEntries(self):
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
        self.open(to_open, opt)

    def open(self, to_open, opt=0):
        if opt == 1:
            self.interval_entry["state"] = tk.NORMAL
            for i in to_open:
                self.model_parameters_frame_options[i][2]["state"] = tk.NORMAL
                self.model_parameters_frame_options[i][3]["state"] = tk.NORMAL
        else:
            for i in to_open:
                self.model_parameters_frame_options[i][1]["state"] = tk.NORMAL

        self.vars_nums = to_open

    def openOtherEntries(self):
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
        if self.seasonal_lookback_option.get():
            self.seasonal_lookback_entry_1["state"] = tk.NORMAL
            self.seasonal_lookback_entry_2["state"] = tk.NORMAL
        else:
            self.seasonal_lookback_entry_1["state"] = tk.DISABLED
            self.seasonal_lookback_entry_2["state"] = tk.DISABLED

    def checkErrors(self):
        try:
            msg = "Read a data first"
            self.df.head(1)

            msg = "Select predictors"
            if not self.predictor_list.get(0):
                raise Exception

            msg = "Select a target"
            if not self.target_list.get(0):
                raise Exception

            msg = "Target and predictor have same variable"
            if self.target_list.get(0) in self.predictor_list.get(0, tk.END):
                raise Exception

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

            # for i, j in enumerate(["Epsilon", "Nu", "C", "Gamma", "Coef0", "Degree"]):
            #    if str(self.model_parameters_frame_options[i][1]["state"]) != "disabled" and not self.parameters[i].get():
            #        msg = "Enter a valid " + j +  " value"
            #        raise Exception

            #    if self.grid_option_var.get():
            #        if str(self.model_parameters_frame_options[i][2]["state"]) != "disabled":
            #            if (not self.optimization_parameters[i][0].get() or not self.optimization_parameters[i][1].get()):
            #                msg = "Enter a valid " + j +  " value in grid search area"
            #                raise Exception

        except Exception:
            popupmsg(msg)  # type: ignore
            return True

    def getLookback(self, X, y, lookback=0, seasons=0, seasonal_lookback=0, sliding=-1):
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

    def getData(self):
        self.is_round = False
        self.is_negative = False
        lookback_option = self.lookback_option.get()
        seasonal_lookback_option = self.seasonal_lookback_option.get()
        sliding = lookback_option + 2 * seasonal_lookback_option - 1
        self.sliding = sliding
        scale_choice = self.scale_var.get()

        self.predictor_names = list(self.predictor_list.get(0, tk.END))
        self.label_name = self.target_list.get(0)

        self.df: pd.DataFrame
        X = self.df[self.predictor_names].copy()
        y = self.df[self.label_name].copy()

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

        try:
            lookback = self.lookback_val_var.get()
        except Exception:
            lookback = 0
        try:
            seasonal_period = self.seasonal_period_var.get()
            seasonal_lookback = self.seasonal_val_var.get()
        except Exception:
            seasonal_period = 0
            seasonal_lookback = 0

        X, y = self.getLookback(
            X, y, lookback, seasonal_period, seasonal_lookback, sliding
        )

        return X, y

    def createModel(self):
        if self.checkErrors():
            return

        do_forecast = self.do_forecast_option.get()
        val_option = self.validation_option.get()

        X, y = self.getData()
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
                    if self.scale_var.get() != "None":
                        pred = self.label_scaler.inverse_transform(pred.reshape(-1, 1)).reshape(-1)  # type: ignore
                        y = self.label_scaler.inverse_transform(y.reshape(-1, 1)).reshape(-1)  # type: ignore
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
                        pred = self.label_scaler.inverse_transform(pred.reshape(-1, 1)).reshape(-1)  # type: ignore
                        y_test = self.label_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)  # type: ignore
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
                    if self.scale_var.get() != "None":
                        pred = self.label_scaler.inverse_transform(pred.reshape(-1, 1)).reshape(-1)  # type: ignore
                        y = self.label_scaler.inverse_transform(y.reshape(-1, 1)).reshape(-1)  # type: ignore
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
                        pred = self.label_scaler.inverse_transform(pred.reshape(-1, 1)).reshape(-1)  # type: ignore
                        y_test = self.label_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)  # type: ignore
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

            p = self.model.get_params()
            self.best_params = {
                "n_estimators": int(p["n_estimators"]),
                "max_depth": int(p["max_depth"]),
                "learning_rate": float(p["learning_rate"]),
            }
            popupmsg("Best Params: " + str(self.best_params))

    def forecastLookback(
        self, num, lookback=0, seasons=0, seasonal_lookback=0, sliding=-1
    ):
        self.test_df: pd.DataFrame
        pred = []
        if sliding == 0:
            last = self.last
            for i in range(num):
                X_test = self.test_df[self.predictor_names].iloc[i]
                if self.scale_var.get() != "None":
                    X_test.iloc[:] = self.feature_scaler.transform(X_test.values.reshape(1, -1)).reshape(-1)  # type: ignore
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
                if self.scale_var.get() != "None":
                    X_test.iloc[:] = self.feature_scaler.transform(X_test.values.reshape(1, -1)).reshape(-1)  # type: ignore
                for j in range(1, seasons + 1):
                    X_test[f"t-{j*seasonal_last}"] = seasonal_last[-j * seasonal_lookback]  # type: ignore
                to_pred = X_test.to_numpy().reshape(1, -1)  # type: ignore
                out = self.model.predict(to_pred)
                seasonal_last = np.append(seasonal_last, out)[1:]
                pred.append(out)

        elif sliding == 2:
            last = self.last
            seasonal_last = self.seasonal_last
            for i in range(num):
                X_test = self.test_df[self.predictor_names].iloc[i]
                if self.scale_var.get() != "None":
                    X_test.iloc[:] = self.feature_scaler.transform(X_test.values.reshape(1, -1)).reshape(-1)  # type: ignore
                for j in range(1, lookback + 1):
                    X_test[f"t-{j}"] = last[-j]  # type: ignore
                for j in range(1, seasons + 1):
                    X_test[f"t-{j*seasonal_lookback}"] = seasonal_last[-j * seasonal_lookback]  # type: ignore
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
        lookback_option = self.lookback_option.get()
        seasonal_lookback_option = self.seasonal_lookback_option.get()
        try:
            X_test = self.test_df[self.predictor_names][:num].to_numpy()  # type: ignore
            y_test = self.test_df[self.label_name][:num].to_numpy().reshape(-1)  # type: ignore
            self.y_test = y_test
        except Exception:
            popupmsg("Read a test data")
            return

        if lookback_option == 0 and seasonal_lookback_option == 0:
            if self.scale_var.get() != "None":
                X_test = self.feature_scaler.transform(X_test)
            self.pred = self.model.predict(X_test).reshape(-1)
        else:
            sliding = self.sliding
            try:
                lookback = self.lookback_val_var.get()
            except Exception:
                lookback = 0
            try:
                seasonal_lookback = self.seasonal_val_var.get()
                seasons = self.seasonal_period_var.get()
            except Exception:
                seasonal_lookback = 0
                seasons = 0

            self.pred = self.forecastLookback(
                num, lookback, seasons, seasonal_lookback, sliding
            )

        if self.scale_var.get() != "None":
            self.pred = self.label_scaler.inverse_transform(self.pred.reshape(-1, 1)).reshape(-1)  # type: ignore

        if not self.is_negative:
            self.pred = self.pred.clip(0, None)
        if self.is_round:
            self.pred = np.round(self.pred).astype(int)

        losses = loss(y_test, self.pred)
        for i in range(len(self.test_metrics_vars)):
            self.test_metrics_vars[i].set(losses[i])

    def vsGraph(self):
        y_test = self.y_test
        try:
            pred = self.pred
        except Exception:
            return
        plt.plot(y_test)
        plt.plot(pred)
        plt.legend(["test", "pred"], loc="upper left")
        plt.show()
