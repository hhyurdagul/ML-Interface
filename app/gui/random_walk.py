import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from pandastable import Table
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import random
from .helpers import *

class RandomWalkRegressor:
    def __init__(self, series, epsilon, seasonal_value=1):
        self.series = series
        self.epsilon = epsilon
        self.seasonal_value = seasonal_value

    def predict(self, n):
        pred = self.series[-self.seasonal_value:].tolist()
        for i in range(n):
            direction = random.choice([self.epsilon, -self.epsilon])
            val = pred[i] + direction
            pred.append(val)

        return np.array(pred[self.seasonal_value:])


class RandomWalk:
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

        self.train_choice = tk.IntVar(value=0)
        tk.Radiobutton(graph_frame, text="As Percent", variable=self.train_choice, value=0).grid(column=0, row=1)
        tk.Radiobutton(graph_frame, text="As Number", variable=self.train_choice, value=1).grid(column=1, row=1)

        lags = tk.IntVar(value=40)
        ttk.Label(graph_frame, text="Lag Number").grid(column=0, row=2)
        ttk.Entry(graph_frame, textvariable=lags).grid(column=1, row=2)

        ttk.Button(graph_frame, text="Show ACF", command=lambda: self.showAcf(lags.get())).grid(column=0, row=3)

        # Crete Model
        create_model_frame = ttk.Labelframe(self.root, text="Create Model")
        create_model_frame.grid(column=1, row=0)

        self.epsilon_var = tk.DoubleVar(value=10)
        ttk.Label(create_model_frame, text="Epsilon Value: ").grid(column=0, row=0)
        ttk.Entry(create_model_frame, textvariable=self.epsilon_var, width=12).grid(column=1, row=0)

        self.seasonal_option = tk.IntVar(value=0)
        tk.Checkbutton(create_model_frame, text="Seasonal", offvalue=0, onvalue=1 ,variable=self.seasonal_option, command=self.openEntries).grid(column=0, row=1, columnspan=2)
        self.seasonal_value = tk.IntVar(value=12)
        ttk.Label(create_model_frame, text="Seasonal Value", width=12).grid(column=0, row=2)
        self.seasonal_value_entry = ttk.Entry(create_model_frame, textvariable=self.seasonal_value, width=12, state=tk.DISABLED)
        self.seasonal_value_entry.grid(column=1, row=2)

        ttk.Button(create_model_frame, text="Create Model", command=self.createModel).grid(column=0, row=5, columnspan=2)

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

        ttk.Button(test_model_main_frame, text="Test Model", command=lambda: self.forecast(self.forecast_num.get())).grid(column=2, row=3)
        ttk.Button(test_model_main_frame, text="Actual vs Forecast Graph", command=self.vsGraph).grid(column=0, row=4, columnspan=3)

        ## Test Model Metrics
        self.test_data_valid = False
        self.forecast_done = False
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
    
    def openEntries(self):
        if self.seasonal_option.get() == 1:
            op = tk.NORMAL
        else:
            op = tk.DISABLED
        self.seasonal_value_entry["state"] = op

    def showAcf(self, lags):
        top = tk.Toplevel()
        fig = plt.Figure((20,15))

        data = self.df[self.target_list.get(0)]
        size = int(self.train_size.get()) if self.train_choice.get() == 1 else int((self.train_size.get()/100)*len(data))
        data = data.iloc[-size:]

        ax = fig.add_subplot(211)
        ax1 = fig.add_subplot(212)

        plot_acf(data, ax=ax, lags=lags)
        plot_pacf(data, ax=ax1, lags=lags)

        canvas = FigureCanvasTkAgg(fig, top)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, top)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)


    def createModel(self):
        self.is_round = False
        self.is_negative = False
        data = self.df[self.target_list.get(0)]
        size = int(self.train_size.get()) if self.train_choice.get() == 1 else int((self.train_size.get()/100)*len(data))
        series = data.iloc[-size:]

        if series.dtype == int or series.dtype == np.intc or series.dtype == np.int64:
            self.is_round = True
        if any(series < 0):
            self.is_negative = True


        seasonal_value = self.seasonal_value.get() if self.seasonal_option.get() else 1
        self.model = RandomWalkRegressor(series, self.epsilon_var.get(), seasonal_value)

    def forecast(self, num):
        self.pred = self.model.predict(num)

        if not self.is_negative:
            self.pred = self.pred.clip(0, None)
        if self.is_round:
            self.pred = np.round(self.pred).astype(int)
        
        if self.test_data_valid:
            y_test = self.test_df[self.target_list.get(0)][:num]
            self.y_test = y_test
            losses = loss(y_test, self.pred)

            for i in range(6):
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
