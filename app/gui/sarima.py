import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from pandastable import Table
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, cross_validate
from sklearn.svm import SVR, NuSVR

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA

from datetime import timedelta
import os
import json

from .helpers import *

class SARIMA:
    def __init__(self):
        self.root = ttk.Frame()
        
        # Get Train Set
        get_train_set_frame = ttk.Labelframe(self.root, text="Get Train Set")
        get_train_set_frame.grid(column=0, row=0)

        file_path = tk.StringVar(value="")
        ttk.Label(get_train_set_frame, text="Train File Path").grid(column=0, row=0)
        ttk.Entry(get_train_set_frame, textvariable=file_path).grid(column=1, row=0)
        ttk.Button(get_train_set_frame, text="Read Csv", command=lambda: self.readCsv(file_path)).grid(column=2, row=0)

        self.input_list = tk.Listbox(get_train_set_frame)
        self.input_list.grid(column=0, row=1)

        self.predictor_list = tk.Listbox(get_train_set_frame)
        self.predictor_list.grid(column=1, row=1)

        self.target_list = tk.Listbox(get_train_set_frame)
        self.target_list.grid(column=2, row=1)

        ttk.Button(get_train_set_frame, text="Add Predictor", command=self.addPredictor).grid(column=1, row=2)
        ttk.Button(get_train_set_frame, text="Eject Predictor", command=self.ejectPredictor).grid(column=1, row=3)

        ttk.Button(get_train_set_frame, text="Add Target", command=self.addTarget).grid(column=2, row=2)
        ttk.Button(get_train_set_frame, text="Eject Target", command=self.ejectTarget).grid(column=2, row=3)
       

        
        # Graphs
        graph_frame = ttk.Labelframe(self.root, text="Graphs")
        graph_frame.grid(column=1, row=0)
        
        ttk.Button(graph_frame, text="Show ACF", command=lambda: self.showAcf(0)).grid(column=0, row=0)
        ttk.Button(graph_frame, text="First Difference ACF", command=lambda: self.showAcf(1)).grid(column=1, row=0)
        
        self.season_number_var = tk.IntVar(value="")
        ttk.Entry(graph_frame, textvariable=self.season_number_var, width=8).grid(column=0, row=1)
        ttk.Button(graph_frame, text="Seasonal Difference ACF", command=lambda: self.showAcf(2)).grid(column=1, row=1)

        ttk.Button(graph_frame, text="Both Difference ACF", command=lambda: self.showAcf(3)).grid(column=0, row=2, columnspan=2)

        # Crete Model
        create_model_frame = ttk.Labelframe(self.root, text="Create Model")
        create_model_frame.grid(column=2, row=0)

        ## Model Without Optimization
        model_without_optimization_frame = ttk.LabelFrame(create_model_frame, text="Model Without Optimization")
        model_without_optimization_frame.grid(column=0, row=0)

        self.pdq_var = [tk.IntVar(value="") for _ in range(3)]
        [
            [
                ttk.Label(model_without_optimization_frame, text=j).grid(column=0, row=i+1),
                ttk.Entry(model_without_optimization_frame, textvariable=self.pdq_var[i], width=8).grid(column=1, row=i+1)
            ] for i, j in enumerate(['p', 'q', 'd'])
        ]
        
        self.seasonality_option = tk.IntVar(value=0)
        tk.Checkbutton(model_without_optimization_frame, text="Seasonality", offvalue=0, onvalue=1, variable=self.seasonality_option, command=self.openSeasons).grid(column=0, row=0, columnspan=2)
        
        self.PQDM_var = [tk.IntVar(value="") for _ in range(4)]
        
        self.seasonals = [
            [
                ttk.Label(model_without_optimization_frame, text=j).grid(column=2, row=i+1),
                ttk.Entry(model_without_optimization_frame, textvariable=self.PQDM_var[i], width=8, state=tk.DISABLED)
            ] for i, j in enumerate(['P', 'Q', 'D', 'M'])
        ]
        
        for i, j in enumerate(self.seasonals):
            j[1].grid(column=3, row=i+1)

        ttk.Button(create_model_frame, text="Create Model", command=self.createModel).grid(column=0, row=1)

        # Test Model
        test_model_frame = ttk.LabelFrame(self.root, text="Test Frame")
        test_model_frame.grid(column=1, row=1, columnspan=2)

        ## Test Model Main
        test_model_main_frame = ttk.LabelFrame(test_model_frame, text="Test Model")
        test_model_main_frame.grid(column=0, row=0)

        forecast_num = tk.IntVar(value="")
        ttk.Label(test_model_main_frame, text="# of Forecast").grid(column=0, row=0)
        ttk.Entry(test_model_main_frame, textvariable=forecast_num).grid(column=1, row=0)
        ttk.Button(test_model_main_frame, text="Values", command=self.showPredicts).grid(column=2, row=0)

        test_file_path = tk.StringVar()
        ttk.Label(test_model_main_frame, text="Test File Path").grid(column=0, row=1)
        ttk.Entry(test_model_main_frame, textvariable=test_file_path).grid(column=1, row=1)
        ttk.Button(test_model_main_frame, text="Get Test Set", command=lambda: self.getTestSet(test_file_path)).grid(column=2, row=1)

        ttk.Button(test_model_main_frame, text="Test Model", command=lambda: self.forecast(forecast_num.get())).grid(column=2, row=3)
        ttk.Button(test_model_main_frame, text="Actual vs Forecast Graph", command=self.vsGraph).grid(column=0, row=4, columnspan=3)

        ## Test Model Metrics
        test_model_metrics_frame = ttk.LabelFrame(test_model_frame, text="Test Metrics")
        test_model_metrics_frame.grid(column=1, row=0)

        test_metrics = ["NMSE", "RMSE", "MAE", "MAPE", "SMAPE"]
        self.test_metrics_vars = [tk.Variable(), tk.Variable(), tk.Variable(), tk.Variable(), tk.Variable(), tk.Variable()]
        for i, j in enumerate(test_metrics):
            ttk.Label(test_model_metrics_frame, text=j).grid(column=0, row=i)
            ttk.Entry(test_model_metrics_frame, textvariable=self.test_metrics_vars[i]).grid(column=1,row=i)

    def readCsv(self, file_path):
        path = filedialog.askopenfilename(filetypes=[("Csv Files", "*.csv"), ("Excel Files", "*.xl*")])
        file_path.set(path)
        if path.endswith(".csv"):
            self.df = pd.read_csv(path)
        else:
            self.df = pd.read_excel(path)
        self.fillInputList()
        
    def fillInputList(self):
        self.input_list.delete(0, tk.END)

        for i in self.df.columns:
            self.input_list.insert(tk.END, i)

    def getTestSet(self, file_path):
        path = filedialog.askopenfilename(filetypes=[("Csv Files", "*.csv"), ("Excel Files", "*.xl*")])
        file_path.set(path)
        if path.endswith(".csv"):
            self.test_df = pd.read_csv(path)
        else:
            self.test_df = pd.read_excel(path)

    def showPredicts(self):
        top = tk.Toplevel(self.root)
        df = pd.DataFrame({"Test": self.y_test, "Predict": self.pred})
        pt = Table(top, dataframe=df, editable=False)
        pt.show()

    def addPredictor(self):
        try:
            a = self.input_list.get(self.input_list.curselection())
            if a not in self.predictor_list.get(0,tk.END):
                self.predictor_list.insert(tk.END, a)
        except:
            pass

    def ejectPredictor(self):
        try:
            self.predictor_list.delete(self.predictor_list.curselection())
        except:
            pass
    
    def addTarget(self):
        try:
            a = self.input_list.get(self.input_list.curselection())
            if self.target_list.size() < 1:
                self.target_list.insert(tk.END, a)
        except:
            pass

    def ejectTarget(self):
        try:
            self.target_list.delete(self.target_list.curselection())
        except:
            pass

    def showAcf(self, choice):
        pass

    def openSeasons(self):
        if self.seasonality_option.get() == 1:
            for i in self.seasonals:
                i[1]["state"] = tk.NORMAL
        else:
            for i in self.seasonals:
                i[1]["state"] = tk.DISABLED

    def createModel(self):
        series = self.df[self.target_list.get(0)].values
        pqd = tuple(i.get() for i in self.pdq_var)

        if self.seasonality_option.get() == 1:
            PQDM = tuple(i.get() for i in self.PQDM_var)
            model = SARIMAX(series, order=pqd, seasonal_order=PQDM)
            self.model = model.fit()
        else:
            model = ARIMA(series, order=pqd)
            self.model = model.fit()
        
        self.end = len(series)

    
    def forecast(self, num):
        
        y_test = self.test_df[self.target_list.get(0)][:num]
        
        self.pred = self.model.predict(start=self.end+1, end=self.end+num)

        self.y_test = y_test

        losses = loss(y_test, self.pred)

        for i in range(6):
            self.test_metrics_vars[i].set(losses[i])

    def vsGraph(self):
        y_test = self.y_test.values.reshape(-1)
        pred = self.pred
        plt.plot(y_test)
        plt.plot(pred)
        plt.legend(["test", "pred"], loc="upper left")
        plt.show()

