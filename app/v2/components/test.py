import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from types import FunctionType
from pandastable import Table

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import json
from typing import List, Tuple, Union

class DatasetInputComponent:
    def __init__(self, root_frame: ttk.Frame, column: int=0, row: int=0):
        self.df: pd.DataFrame
        self.train_file_path_var: tk.StringVar

        train_set_frame = self.attach_frame(root_frame, column, row)
        self.attach_input_list(train_set_frame)

    def _add_predictor(self, _=None):
        current_selection = self.input_list.curselection()
        if current_selection is not ():
            a = self.input_list.get(current_selection)
            if a not in self.predictor_list.get(0,tk.END):
                self.predictor_list.insert(tk.END, a)
    
    def _eject_predictor(self, _=None):
        current_selection = self.predictor_list.curselection()
        if current_selection is not ():
            self.predictor_list.delete(current_selection)
    
    def _add_target(self, _=None):
        current_selection = self.input_list.curselection()
        if current_selection is not () and self.target_list.size() < 1:
            a = self.input_list.get(current_selection)
            self.target_list.insert(tk.END, a)

    def _eject_target(self, _=None):
        current_selection = self.target_list.curselection()
        if current_selection is not ():
            self.target_list.delete(current_selection)
        
    def _fill_input_list(self):
        self.input_list.delete(0, tk.END)

        for i in self.df.columns.to_list():
            self.input_list.insert(tk.END, i)

    def _read_csv(self):
        path = filedialog.askopenfilename(filetypes=[("Csv Files", "*.csv"), ("Excel Files", "*.xlsx"), ("Xls Files", "*.xls")])
        if not path:
            return
        self.train_file_path_var.set(path)
        if path.endswith(".csv"):
            self.df = pd.read_csv(path)
        else:
            self.df = pd.read_excel(path, engine="openpyxl")
        self._fill_input_list()

    def get_predictors(self) -> list:
        return list(self.predictor_list.get(0, tk.END))

    def get_label(self) -> str:
        return self.target_list.get(0)

    def get_params(self) -> dict:
        return {"predictor_names": self.get_predictors(), "label_name": self.get_label()}

    def check_errors(self) -> Tuple[bool, str]:
        if not self.train_file_path_var.get():
            return (True, "Read a data first")
        if not self.predictor_list.get(0):
            return (True, "Select predictors")
        if not self.target_list.get(0):
            return (True, "Select a target")
        if self.get_label() in self.get_predictors():
            return (True, "Target and predictor have same variable")
        return (False, "")

    def attach_frame(self, root_frame: ttk.Frame, column: int, row: int) -> ttk.Labelframe:
        # Get Train Set
        train_set_frame = ttk.Labelframe(root_frame, text="Get Train Set")
        train_set_frame.grid(column=0, row=0)

        self.train_file_path_var = tk.StringVar(value="")
        ttk.Label(train_set_frame, text="Train File Path").grid(column=column, row=row)
        ttk.Entry(train_set_frame, textvariable=self.train_file_path_var).grid(column=1, row=0)
        ttk.Button(train_set_frame, text="Read Data", command=self._read_csv).grid(column=2, row=0)

        return train_set_frame

    def attach_input_list(self, root_frame: Union[ttk.Frame, ttk.LabelFrame]):
        self.input_list = tk.Listbox(root_frame)
        self.input_list.grid(column=0, row=1)
        self.input_list.bind("<Double-Button-1>", self._add_predictor)
        self.input_list.bind("<Double-Button-3>", self._add_target)

        self.predictor_list = tk.Listbox(root_frame)
        self.predictor_list.grid(column=1, row=1)
        self.predictor_list.bind("<Double-Button-1>", self._eject_predictor)

        self.target_list = tk.Listbox(root_frame)
        self.target_list.grid(column=2, row=1)
        self.target_list.bind("<Double-Button-1>", self._eject_target)

        ttk.Button(root_frame, text="Add Predictor", command=self._add_predictor).grid(column=1, row=2)
        ttk.Button(root_frame, text="Eject Predictor", command=self._eject_predictor).grid(column=1, row=3)
        ttk.Button(root_frame, text="Debug", command=lambda: print(self.input_list.get(0))).grid(column=0, row=2)
        ttk.Button(root_frame, text="Debug 2", command=lambda: print(self.input_list.curselection())).grid(column=0, row=3)

        ttk.Button(root_frame, text="Add Target", command=self._add_target).grid(column=2, row=2)
        ttk.Button(root_frame, text="Eject Target", command=self._eject_target).grid(column=2, row=3)

class ModelValidationComponent:
    def __init__(self, root_frame: ttk.Frame, column: int=0, row: int=1):
        self.do_forecast_option_var: tk.IntVar
        self.validation_option_var: tk.IntVar
        self.random_percent_var: tk.IntVar
        self.k_fold_value_var: tk.IntVar
        
        self.attach_frame(root_frame, column, row)
    
    def _open_entries(self):
        if self.do_forecast_option_var.get():
            self.entry_vars[0]["state"] = tk.NORMAL if self.validation_option_var.get() else tk.DISABLED
            for radiobutton in self.radio_vars:
                radiobutton["state"] = tk.DISABLED
        else: 
            self.entry_vars[0]["state"] = tk.NORMAL if self.validation_option_var.get() == 1 else tk.DISABLED
            self.entry_vars[1]["state"] = tk.NORMAL if self.validation_option_var.get() == 2 else tk.DISABLED
            for radiobutton in self.radio_vars:
                radiobutton["state"] = tk.NORMAL
    
    def set_params(self, params: dict):
        self.do_forecast_option_var.set(params["do_forecast"])
        self.validation_option_var.set(params["validation_option"])
        if params["validation_option"] == 1:
            self.random_percent_var.set(params["random_percent"])
        elif params["validation_option"] == 2:
            self.k_fold_value_var.set(params["k_fold_cv"])

    def get_params(self) -> dict:
        return {
            "do_forecast": self.do_forecast_option_var.get(),
            "validation_option": self.validation_option_var.get(),
            "random_percent": self.random_percent_var.get() if self.validation_option_var.get() == 1 else None,
            "k_fold_cv": self.k_fold_value_var.get() if self.validation_option_var.get() == 2 else None,
        }
    
    def check_errors(self) -> Tuple[bool, str]:
        if self.random_percent_var.get() <= 0:
            return (True, "Enter a valid percent value")
        if self.validation_option_var.get() == 2 and self.k_fold_value_var.get() <= 1:
            return (True, "Enter a valid K-fold value (Above 2)")
        return (False, "")


    def attach_frame(self, root_frame: ttk.Frame, column: int, row: int):
        # Model testing and validation
        model_validation_frame = ttk.Labelframe(root_frame, text="Model testing and validation")
        model_validation_frame.grid(column=column, row=row)

        self.do_forecast_option_var = tk.IntVar(value=0)
        tk.Checkbutton(model_validation_frame, text="Do Forecast", offvalue=0, onvalue=1, variable=self.do_forecast_option_var, command=self._open_entries).grid(column=0, row=0, columnspan=2)
        
        self.validation_option_var = tk.IntVar(value=0)
        tk.Radiobutton(model_validation_frame, text="No validation, use all data rows", value=0, variable=self.validation_option_var, command=self._open_entries).grid(column=0, row=1, columnspan=2, sticky=tk.W)
        
        self.random_percent_var = tk.IntVar(value=70)
        tk.Radiobutton(model_validation_frame, text="Random percent", value=1, variable=self.validation_option_var, command=self._open_entries).grid(column=0, row=2, sticky=tk.W)
        random_percent_value_entry = ttk.Entry(model_validation_frame, textvariable=self.random_percent_var, width=8)
        random_percent_value_entry.grid(column=1, row=2)

        k_fold_radio = tk.Radiobutton(model_validation_frame, text="K-fold cross-validation", value=2, variable=self.validation_option_var, command=self._open_entries)
        k_fold_radio.grid(column=0, row=3, sticky=tk.W)
        leave_one_out_radio = tk.Radiobutton(model_validation_frame, text="Leave one out cross-validation", value=3, variable=self.validation_option_var, command=self._open_entries)
        leave_one_out_radio.grid(column=0, row=4, columnspan=2, sticky=tk.W)

        self.k_fold_value_var = tk.IntVar(value=5)
        k_fold_value_entry = ttk.Entry(model_validation_frame, textvariable=self.k_fold_value_var, width=8)
        k_fold_value_entry.grid(column=1, row=3)

        self.radio_vars = [k_fold_radio, leave_one_out_radio]
        self.entry_vars = [random_percent_value_entry, k_fold_value_entry]

class CustomizeTrainsetComponent:
    def __init__(self, root_frame: ttk.Frame, column: int=0, row: int=2):
        self.scale_var: tk.StringVar
        self.lookback_attached = False
        self.attach_frame(root_frame, column, row)

    def attach_frame(self, root_frame: ttk.Frame, column: int, row: int):
        # Customize Train Set
        self.customize_train_set_frame = ttk.LabelFrame(root_frame, text="Customize Train Set")
        self.customize_train_set_frame.grid(column=column, row=row)
        
        self.scale_var = tk.StringVar(value="None")
        ttk.Label(self.customize_train_set_frame, text="Scale Type").grid(column=0, row=0)
        ttk.OptionMenu(self.customize_train_set_frame, self.scale_var, "None", "None","StandardScaler", "MinMaxScaler").grid(column=1, row=0)

    def _open_lookback_entries(self):
        self.entry_vars[0]["state"] = tk.NORMAL if self.lookback_option_var.get() else tk.DISABLED
        self.entry_vars[1]["state"] = tk.NORMAL if self.seasonal_lookback_option_var.get() else tk.DISABLED
        self.entry_vars[2]["state"] = tk.NORMAL if self.seasonal_lookback_option_var.get() else tk.DISABLED
    
    def get_sliding(self) -> int:
        return self.lookback_option_var.get() + 2 * self.seasonal_lookback_option_var.get() - 1;

    def set_params(self, params: dict):
        self.scale_var.set(params["scale_type"])
        if self.lookback_attached:
            self.lookback_option_var.set(params["lookback_option"]) 
            self.lookback_value_var.set(params["lookback_value"])
            self.seasonal_lookback_option_var.set(params["seasonal_lookback_option"]) 
            self.seasonal_period_value_var.set(params["seasonal_period"])
            self.seasonal_value_var.set(params["seasonal_value"])

    def get_params(self) -> dict:
        params = {"scale_type": self.scale_var.get()}
        if self.lookback_attached:
            lookback_params = {
                "lookback_option": self.lookback_option_var.get(),
                "lookback_value": self.lookback_value_var.get() if self.lookback_option_var.get() else None,
                "seasonal_lookback_option": self.seasonal_lookback_option_var.get(),
                "seasonal_period": self.seasonal_period_value_var.get() if self.seasonal_lookback_option_var.get() else None,
                "seasonal_value": self.seasonal_value_var.get() if self.seasonal_lookback_option_var.get() else None,
            }
            params = dict(list(params.items()) + list(lookback_params.items()))
        return params 
    
    def check_errors(self) -> Tuple[bool, str]:
        if not self.lookback_attached:
            return (False, "")
        if self.lookback_option_var.get() and not self.lookback_value_var.get():
            return (True, "Enter a valid lookback value")
        if self.seasonal_lookback_option_var.get():
            if not self.seasonal_value_var.get() or not self.seasonal_period_value_var.get():
                return (True, "Select predictors")
        return (False, "")

    def attach_lookback(self):
        self.lookback_option_var = tk.IntVar(value=0)
        self.lookback_value_var = tk.IntVar(value=0)
        tk.Checkbutton(self.customize_train_set_frame, text="Lookback", offvalue=0, onvalue=1, variable=self.lookback_option_var, command=self._open_lookback_entries).grid(column=0, row=1)

        lookback_entry = tk.Entry(self.customize_train_set_frame, textvariable=self.lookback_value_var, width=8, state=tk.DISABLED)
        lookback_entry.grid(column=1, row=1)

        self.seasonal_lookback_option_var = tk.IntVar(value=0)
        self.seasonal_period_value_var = tk.IntVar(value=0)
        self.seasonal_value_var = tk.IntVar(value=0)
        tk.Checkbutton(self.customize_train_set_frame, text="Periodic Lookback", offvalue=0, onvalue=1, variable=self.seasonal_lookback_option_var, command=self._open_lookback_entries).grid(column=0, row=2)

        seasonal_lookback_entry_1 = tk.Entry(self.customize_train_set_frame, textvariable=self.seasonal_period_value_var, width=9, state=tk.DISABLED)
        seasonal_lookback_entry_1.grid(column=0, row=3)
        seasonal_lookback_entry_2 = tk.Entry(self.customize_train_set_frame, textvariable=self.seasonal_value_var, width=8, state=tk.DISABLED)
        seasonal_lookback_entry_2.grid(column=1, row=3)

        self.entry_vars = [lookback_entry, seasonal_lookback_entry_1, seasonal_lookback_entry_2]
        self.lookback_attached = True

class TestModelComponent:
    def __init__(self, root_frame: ttk.Frame, column: int=1, row: int=1, **functions):
        self.test_file_path_var: tk.StringVar
        self.forecast_num_var: tk.IntVar
        self.test_metrics_vars: List[tk.DoubleVar]

        self.forecast_function = functions["forecast_function"]
        self.graph_predicts_function = functions["graph_predicts_function"]
        self.show_predict_values_function = functions["show_predict_values_function"]

        self.type = "Supervised"
        self.attach_frame(root_frame, column, row)

    def _read_test_set(self):
        path = filedialog.askopenfilename(filetypes=[("Csv Files", "*.csv"), ("Excel Files", "*.xlsx"), ("Xls Files", "*.xls")])
        if not path:
            return
        self.test_file_path_var.set(path)
        if path.endswith(".csv"):
            self.df = pd.read_csv(path)
        else:
            self.df = pd.read_excel(path, engine="openpyxl")

    def check_errors(self) -> Tuple[bool, str]:
        if self.forecast_num_var.get() <= 0:
            return (True, "Enter a valid forecast count")
        if self.type != "Timeseries" and self.test_file_path_var.get() == "":
            return (True, "Enter a valid test dataset")
        return (False, "")

    def attach_frame(self, root_frame: ttk.Frame, column: int, row: int):
        test_model_frame = ttk.LabelFrame(root_frame, text="Test Frame")
        test_model_frame.grid(column=column, row=row)

        ## Test Model Main
        test_model_main_frame = ttk.LabelFrame(test_model_frame, text="Test Model")
        test_model_main_frame.grid(column=0, row=0)

        self.forecast_num_var = tk.IntVar(value=7)
        ttk.Label(test_model_main_frame, text="# of Forecast").grid(column=0, row=0)
        ttk.Entry(test_model_main_frame, textvariable=self.forecast_num_var).grid(column=1, row=0)
        ttk.Button(test_model_main_frame, text="Values", command=self.show_predict_values_function).grid(column=2, row=0)

        self.test_file_path_var = tk.StringVar()
        ttk.Label(test_model_main_frame, text="Test File Path").grid(column=0, row=1)
        ttk.Entry(test_model_main_frame, textvariable=self.test_file_path_var).grid(column=1, row=1)
        ttk.Button(test_model_main_frame, text="Get Test Set", command=self._read_test_set).grid(column=2, row=1)

        ttk.Button(test_model_main_frame, text="Test Model", command=self.forecast_function).grid(column=2, row=3)
        ttk.Button(test_model_main_frame, text="Actual vs Forecast Graph", command=self.graph_predicts_function).grid(column=0, row=4, columnspan=3)

        ## Test Model Metrics
        test_model_metrics_frame = ttk.LabelFrame(test_model_frame, text="Test Metrics")
        test_model_metrics_frame.grid(column=1, row=0)

        test_metrics = ["NMSE", "RMSE", "MAE", "MAPE", "SMAPE"]
        self.test_metrics_vars = [tk.DoubleVar() for _ in range(len(test_metrics))]
        for i, j in enumerate(test_metrics):
            ttk.Label(test_model_metrics_frame, text=j).grid(column=0, row=i)
            ttk.Entry(test_model_metrics_frame, textvariable=self.test_metrics_vars[i]).grid(column=1, row=i)
    
