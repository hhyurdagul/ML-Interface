import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from pandastable import Table
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

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
        ttk.Button(get_train_set_frame, text="Read Data", command=lambda: self.readCsv(file_path)).grid(column=2, row=0)

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

        ttk.Button(get_train_set_frame, text="Add Predictor", command=self.addPredictor).grid(column=1, row=2)
        ttk.Button(get_train_set_frame, text="Eject Predictor", command=self.ejectPredictor).grid(column=1, row=3)

        ttk.Button(get_train_set_frame, text="Add Target", command=self.addTarget).grid(column=2, row=2)
        ttk.Button(get_train_set_frame, text="Eject Target", command=self.ejectTarget).grid(column=2, row=3)
        
        # Graphs
        graph_frame = ttk.Labelframe(self.root, text="Graphs")
        graph_frame.grid(column=0, row=1)
        
        self.train_size = tk.IntVar(value=100)
        ttk.Label(graph_frame, text="Train Size").grid(column=0, row=0)
        ttk.Entry(graph_frame, textvariable=self.train_size).grid(column=1, row=0)

        self.train_size_choice = tk.IntVar(value=0)
        tk.Radiobutton(graph_frame, text="As Percent", variable=self.train_size_choice, value=0).grid(column=0, row=1)
        tk.Radiobutton(graph_frame, text="As Number", variable=self.train_size_choice, value=1).grid(column=1, row=1)

        lags = tk.IntVar(value=40)
        ttk.Label(graph_frame, text="Lag Number").grid(column=0, row=2)
        ttk.Entry(graph_frame, textvariable=lags).grid(column=1, row=2)

        ttk.Button(graph_frame, text="Show ACF", command=lambda: self.showAcf(0, lags.get())).grid(column=0, row=3)
        ttk.Button(graph_frame, text="First Difference ACF", command=lambda: self.showAcf(1, lags.get())).grid(column=1, row=3)
        
        self.season_number_var = tk.IntVar(value="")
        ttk.Entry(graph_frame, textvariable=self.season_number_var, width=8).grid(column=0, row=4)
        ttk.Button(graph_frame, text="Seasonal Difference ACF", command=lambda: self.showAcf(2, lags.get())).grid(column=1, row=4)

        ttk.Button(graph_frame, text="Both Difference ACF", command=lambda: self.showAcf(3, lags.get())).grid(column=0, row=5, columnspan=2)

        # Crete Model
        create_model_frame = ttk.Labelframe(self.root, text="Create Model")
        create_model_frame.grid(column=1, row=0)

        ## Model Without Optimization
        model_without_optimization_frame = ttk.LabelFrame(create_model_frame, text="Model Without Optimization")
        model_without_optimization_frame.grid(column=0, row=0)

        self.pdq_var = [tk.IntVar(value="") for _ in range(3)]
        [
            [
                ttk.Label(model_without_optimization_frame, text=j).grid(column=0, row=i+1),
                ttk.Entry(model_without_optimization_frame, textvariable=self.pdq_var[i], width=8).grid(column=1, row=i+1)
            ] for i, j in enumerate(['p', 'd', 'q'])
        ]
        
        self.seasonality_option = tk.IntVar(value=0)
        tk.Checkbutton(model_without_optimization_frame, text="Seasonality", offvalue=0, onvalue=1, variable=self.seasonality_option, command=self.openSeasons).grid(column=0, row=0, columnspan=2)
        
        self.PDQM_var = [tk.IntVar(value="") for _ in range(4)]
        
        self.seasonals = [
            [
                ttk.Label(model_without_optimization_frame, text=j).grid(column=2, row=i+1),
                ttk.Entry(model_without_optimization_frame, textvariable=self.PDQM_var[i], width=8, state=tk.DISABLED)
            ] for i, j in enumerate(['P', 'D', 'Q', 's'])
        ]
        
        for i, j in enumerate(self.seasonals):
            j[1].grid(column=3, row=i+1)

        ttk.Button(create_model_frame, text="Create Model", command=self.createModel).grid(column=0, row=1)
        ttk.Button(create_model_frame, text="Save Model", command=self.saveModel).grid(column=1, row=1)

        # Test Model
        test_model_frame = ttk.LabelFrame(self.root, text="Test Frame")
        test_model_frame.grid(column=1, row=1)

        ## Test Model Main
        test_model_main_frame = ttk.LabelFrame(test_model_frame, text="Test Model")
        test_model_main_frame.grid(column=0, row=0)

        self.forecast_num = tk.IntVar(value="")
        ttk.Label(test_model_main_frame, text="# of Forecast").grid(column=0, row=0)
        ttk.Entry(test_model_main_frame, textvariable=self.forecast_num).grid(column=1, row=0)
        ttk.Button(test_model_main_frame, text="Values", command=self.showPredicts).grid(column=2, row=0)

        test_file_path = tk.StringVar()
        ttk.Label(test_model_main_frame, text="Test File Path").grid(column=0, row=1)
        ttk.Entry(test_model_main_frame, textvariable=test_file_path).grid(column=1, row=1)
        ttk.Button(test_model_main_frame, text="Get Test Set", command=lambda: self.getTestSet(test_file_path)).grid(column=2, row=1)

        ttk.Button(test_model_main_frame, text="Load Model", command=self.loadModel).grid(column=0, row=3)
        ttk.Button(test_model_main_frame, text="Forecast", command=lambda: self.forecast(self.forecast_num.get())).grid(column=2, row=3)
        ttk.Button(test_model_main_frame, text="Actual vs Forecast Graph", command=self.vsGraph).grid(column=0, row=4, columnspan=3)

        ## Test Model Metrics
        self.test_data_valid = False
        self.forecast_done = False
        test_model_metrics_frame = ttk.LabelFrame(test_model_frame, text="Test Metrics")
        test_model_metrics_frame.grid(column=1, row=0)

        test_metrics = ["NMSE", "RMSE", "MAE", "MAPE", "SMAPE"]
        self.test_metrics_vars = [tk.Variable() for _ in range(len(test_metrics))]
        for i, j in enumerate(test_metrics):
            ttk.Label(test_model_metrics_frame, text=j).grid(column=0, row=i)
            ttk.Entry(test_model_metrics_frame, textvariable=self.test_metrics_vars[i]).grid(column=1,row=i)

    def readCsv(self, file_path):
        path = filedialog.askopenfilename(filetypes=[("Csv Files", "*.csv"), ("Xlsx Files", "*.xlsx"), ("Xlrd Files", ".xls")])
        file_path.set(path)
        if path.endswith(".csv"):
            self.df = pd.read_csv(path)
        else:
            try:
                self.df = pd.read_excel(path)
            except:
                self.df = pd.read_excel(path, engine="openpyxl")
        self.fillInputList()
        
    def fillInputList(self):
        self.input_list.delete(0, tk.END)

        for i in self.df.columns:
            self.input_list.insert(tk.END, i)

    def getTestSet(self, file_path):
        path = filedialog.askopenfilename(filetypes=[("Csv Files", "*.csv"), ("Xlsx Files", "*.xlsx"), ("Xlrd Files", ".xls")])
        file_path.set(path)
        if path.endswith(".csv"):
            self.test_df = pd.read_csv(path)
        else:
            try:
                self.test_df = pd.read_excel(path)
            except:
                self.test_df = pd.read_excel(path, engine="openpyxl")
        self.test_data_valid = True
        if self.forecast_done:
            self.forecast(self.forecast_num.get())

    def showPredicts(self):
        d = {}
        if self.test_data_valid:
            self.y_test: np.ndarray
            d["Test"] = self.y_test
        self.pred: np.ndarray
        try:
            d["Predict"] = self.pred
        except:
            return
        df = pd.DataFrame(d)
        top = tk.Toplevel(self.root)
        pt = Table(top, dataframe=df, editable=False)
        pt.show()

    def addPredictor(self, _=None):
        try:
            a = self.input_list.get(self.input_list.curselection())
            if a not in self.predictor_list.get(0,tk.END):
                self.predictor_list.insert(tk.END, a)
        except:
            pass

    def ejectPredictor(self, _=None):
        try:
            self.predictor_list.delete(self.predictor_list.curselection())
        except:
            pass
    
    def addTarget(self, _=None):
        try:
            a = self.input_list.get(self.input_list.curselection())
            if self.target_list.size() < 1:
                self.target_list.insert(tk.END, a)
        except:
            pass

    def ejectTarget(self, _=None):
        try:
            self.target_list.delete(self.target_list.curselection())
        except:
            pass
    
    def saveModel(self):
        path = filedialog.asksaveasfilename()
        if not path:
            return
        params = {
            "p": self.pdq_var[0].get(),
            "d": self.pdq_var[1].get(),
            "q": self.pdq_var[2].get()
        }

        if self.seasonality_option.get():
            params["P"] = self.PDQM_var[0].get()
            params["D"] = self.PDQM_var[1].get()
            params["Q"] = self.PDQM_var[2].get()
            params["s"] = self.PDQM_var[3].get()

        params["predictor_names"] = self.predictor_names
        params["label_name"] = self.label_name
        params["is_round"] = self.is_round
        params["is_negative"] = self.is_negative
        params["train_size"] = self.train_size.get()
        params["train_size_choice"] = self.train_size_choice.get()
        params["seasonality_option"] = self.seasonality_option.get()
        params["end_len"] = self.end
        os.mkdir(path)

        with open(path+"/model", 'wb') as model_path:
            self.model.save(model_path)
        with open(path+"/model.json", 'w') as outfile:
            json.dump(params, outfile)

    def loadModel(self):
        path = filedialog.askdirectory()
        if not path:
            return
        try:
            model_path = path + "/model"
        except:
            popupmsg("There is no model file at the path")
            return
        with open(path+"/model", 'rb') as model_path:
            self.model = ARIMAResults.load(model_path)
        infile = open(path+"/model.json")
        params = json.load(infile)

        self.pdq_var[0].set(self.model.model_orders["ar"])
        self.pdq_var[2].set(self.model.model_orders["ma"])
        self.pdq_var[1].set(params["d"])

        self.seasonality_option.set(params["seasonality_option"])
        if self.seasonality_option.get():
            self.PDQM_var[0].set(params["P"])
            self.PDQM_var[1].set(params["D"])
            self.PDQM_var[2].set(params["Q"])
            self.PDQM_var[3].set(params["s"])
        
        self.predictor_names = params["predictor_names"]
        self.label_name = params["label_name"]
        self.end = params["end_len"]
        try:
            self.is_round = params["is_round"]
        except:
            self.is_round = True
        try:
            self.is_negative = params["is_negative"]
        except:
            self.is_negative = False
        
        self.train_size.set(params["train_size"])
        self.train_size_choice.set(params["train_size_choice"])

        msg = f"Predictor names are {self.predictor_names}\nLabel name is {self.label_name}"
        popupmsg(msg)

    def showAcf(self, choice, lags):
        top = tk.Toplevel()
        fig = plt.Figure((20,15))

        data = self.df[self.target_list.get(0)]
        size = int(self.train_size.get()) if self.train_size_choice.get() == 1 else int((self.train_size.get()/100)*len(data))
        data = data.iloc[-size:]

        ax = fig.add_subplot(211)
        ax1 = fig.add_subplot(212)

        if choice == 0:
            plot_acf(data, ax=ax, lags=lags)
            plot_pacf(data, ax=ax1, lags=lags)
        elif choice == 1:
            plot_acf(data.diff()[1:], ax=ax, lags=lags)
            plot_pacf(data.diff()[1:], ax=ax1, lags=lags)
        elif choice == 2:
            s = self.season_number_var.get()
            plot_acf(data.diff(s)[s:], ax=ax, lags=lags)
            plot_pacf(data.diff(s)[s:], ax=ax1, lags=lags)
        elif choice == 3:
            s = self.season_number_var.get()
            plot_acf(data.diff()[1:].diff(s)[s:], ax=ax, lags=lags)
            plot_pacf(data.diff()[1:].diff(s)[s:], ax=ax1, lags=lags)

        canvas = FigureCanvasTkAgg(fig, top)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, top)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def openSeasons(self):
        if self.seasonality_option.get() == 1:
            for i in self.seasonals:
                i[1]["state"] = tk.NORMAL
        else:
            for i in self.seasonals:
                i[1]["state"] = tk.DISABLED

    def createModel(self):
        self.is_round = False
        self.is_negative = False
        self.predictor_names = self.predictor_list.get(0)
        self.label_name = self.target_list.get(0)
        data = self.df[self.label_name]
        size = int(self.train_size.get()) if self.train_size_choice.get() == 1 else int((self.train_size.get()/100)*len(data))
        series = data.iloc[-size:]

        if series.dtype == int or series.dtype == np.intc or series.dtype == np.int64:
            self.is_round = True
        if any(series < 0):
            self.is_negative = True

        pqd = tuple(i.get() for i in self.pdq_var)
        
        if self.seasonality_option.get() == 1:
            PDQM = tuple(i.get() for i in self.PDQM_var)
            model = ARIMA(series, order=pqd, seasonal_order=PDQM)
            self.model = model.fit()
            self.end = len(series)
        else:
            model = ARIMA(series, order=pqd)
            self.model = model.fit()
            self.end = len(series)-1
    
    def forecast(self, num):
        self.pred = self.model.predict(start=self.end+1, end=self.end+num)

        if not self.is_negative:
            self.pred = self.pred.clip(0, None)
        if self.is_round:
            self.pred = np.round(self.pred).astype(int)

        self.forecast_done = True

        if self.test_data_valid:
            y_test = self.test_df[self.label_name][:num]
            self.pred.index = y_test.index
            self.y_test = y_test
            losses = loss(y_test, self.pred)
            for i in range(len(self.test_metrics_vars)):
                self.test_metrics_vars[i].set(losses[i])

    def vsGraph(self):
        if self.test_data_valid:
            plt.plot(self.y_test, label="Test")
        try:
            plt.plot(self.pred, label="Predict")
        except:
            return
        plt.legend(loc="upper left")
        plt.show()

