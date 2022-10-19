import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandastable import Table
from .helpers import loss

class MovingAverage:
    def __init__(self):
        self.root = ttk.Frame()

        train_frame = ttk.LabelFrame(self.root, text="Train")
        train_frame.grid(column=0, row=0)

        file_path = tk.StringVar(value="")
        ttk.Label(train_frame, text="Train File Path").grid(column=0, row=0)
        ttk.Entry(train_frame, textvariable=file_path).grid(column=1, row=0)
        ttk.Button(train_frame, text="Read Train Data", command=lambda: self.readTrainData(file_path)).grid(column=2, row=0)

        self.input_list = tk.Listbox(train_frame)
        self.input_list.grid(column=1, row=1)
        self.col_var = tk.StringVar(value="Select Column")
        ttk.Button(train_frame, textvariable=self.col_var, command=self.selectColumn).grid(column=2, row=1)

        self.train_size = tk.IntVar(value=10)
        ttk.Label(train_frame, text="Train Count").grid(column=0, row=2)
        ttk.Entry(train_frame, textvariable=self.train_size).grid(column=1, row=2)
        
        self.period_size = tk.IntVar(value=48)
        ttk.Label(train_frame, text="Period Size").grid(column=0, row=4)
        ttk.Entry(train_frame, textvariable=self.period_size).grid(column=1, row=4)
        
        self.period_count = tk.IntVar(value=7)
        ttk.Label(train_frame, text="Period Count").grid(column=0, row=5)
        ttk.Entry(train_frame, textvariable=self.period_count).grid(column=1, row=5)

        test_frame = ttk.LabelFrame(self.root, text="Test")
        test_frame.grid(column=0, row=1)
        
        self.forecast_num = tk.IntVar(value="") # type: ignore
        ttk.Label(test_frame, text="# of Forecast").grid(column=0, row=0)
        ttk.Entry(test_frame, textvariable=self.forecast_num).grid(column=1, row=0)

        test_file_path = tk.StringVar(value="")
        ttk.Label(test_frame, text="Test File Path").grid(column=0, row=1)
        ttk.Entry(test_frame, textvariable=test_file_path).grid(column=1, row=1)
        ttk.Button(test_frame, text="Read Test Data", command=lambda: self.readTestData(test_file_path)).grid(column=2, row=1)

        ttk.Button(test_frame, text="Create Model", command=self.createModel).grid(column=0, row=2)
        ttk.Button(test_frame, text="Graph Data", command=self.graphData).grid(column=1, row=2)
        ttk.Button(test_frame, text="Show Data", command=self.showData).grid(column=2, row=2)


        self.do_test = 0
        test_metrics = ["NMSE", "RMSE", "MAE", "MAPE", "SMAPE"]
        self.test_metrics_vars = [tk.Variable() for _ in range(len(test_metrics))]
        for i, j in enumerate(test_metrics):
            ttk.Label(test_frame, text=j).grid(column=3, row=i)
            ttk.Entry(test_frame, textvariable=self.test_metrics_vars[i], width=8).grid(column=4, row=i)


    def readTrainData(self, file_path):
        path = filedialog.askopenfilename(filetypes=[("Csv Files", "*.csv"), ("Excel Files", "*.xl*")])
        file_path.set(path)
        if path.endswith(".csv"):
            self.df = pd.read_csv(path)
        else:
            self.df = pd.read_excel(path, index_col=0, engine="openpyxl")

        self.input_list.delete(0, tk.END)
        for i in self.df.columns.tolist(): # type: ignore
            self.input_list.insert(tk.END, i)
    
    def readTestData(self, file_path):
        path = filedialog.askopenfilename(filetypes=[("Csv Files", "*.csv"), ("Excel Files", "*.xl*")])
        file_path.set(path)
        if path.endswith(".csv"):
            self.test_df = pd.read_csv(path)
        else:
            self.test_df = pd.read_excel(path, index_col=0, engine="openpyxl")
        self.do_test = 1

    def selectColumn(self):
        a = self.input_list.get(self.input_list.curselection())
        self.col_var.set("Selected " + a)
        self.col = a

    def graphData(self):
        if self.do_test:
            y_test = self.test_df[self.col].iloc[:self.forecast_num.get()].values # type: ignore
            plt.plot(y_test, label="Test")
        plt.plot(self.pred, label="Pred")
        plt.legend()
        plt.show()

    def showData(self):
        d = {"Predict": self.pred}
        if self.do_test:
            y_test = self.test_df[self.col].iloc[:self.forecast_num.get()].values # type: ignore
            d["Test"] = y_test
        df = pd.DataFrame(d)
        top = tk.Toplevel(self.root)
        pt = Table(top, dataframe=df, editable=False)
        pt.show()

    def createModel(self):
        period = self.period_count.get() * self.period_size.get()
        train_size = period * self.train_size.get()
        
        train = self.df[self.col].values # type: ignore
        val = train.copy()
        
        pred = []
        for _ in range(self.forecast_num.get()//period+1):
            val = val[-train_size:]
            out = np.array([val[i::period].mean() for i in range(period)]) # type: ignore
            val = np.append(val, out)
            pred.extend(out.tolist())

        self.pred = np.array(pred)[:self.forecast_num.get()]
        print(self.pred)
        if self.do_test:
            y_test = self.test_df[self.col].iloc[:self.forecast_num.get()].values # type: ignore
            losses = loss(y_test, self.pred)
            for i in range(6):
                self.test_metrics_vars[i].set(losses[i])
