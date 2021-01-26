import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import matplotlib.pyplot as plt
from pandastable import Table

import pandas as pd
import numpy as np
from scipy.stats import norm # type: ignore
from sklearn.metrics import mean_absolute_error as mae

from .helpers import loss

from random import seed
from numpy.random import seed as np_seed # type: ignore
seed(0)
np_seed(0)

class MonteCarlo:
    def __init__(self):
        self.root = ttk.Frame()

        # Get Train Set
        train_frame = ttk.Labelframe(self.root, text="Train")
        train_frame.grid(column=0, row=0)

        file_path = tk.StringVar(value="")
        ttk.Label(train_frame, text="Train File Path").grid(column=0, row=0)
        ttk.Entry(train_frame, textvariable=file_path).grid(column=1, row=0)
        ttk.Button(train_frame, text="Read Train Data", command=lambda: self.readTrainData(file_path)).grid(column=2, row=0)

        self.input_list = tk.Listbox(train_frame)
        self.input_list.grid(column=1, row=1)
        self.col_var = tk.StringVar(value="Select Column")
        ttk.Button(train_frame, textvariable=self.col_var, command=self.selectColumn).grid(column=2, row=1)
       
        self.train_size = tk.IntVar(value=4)
        ttk.Label(train_frame, text="Train Count").grid(column=0, row=2)
        ttk.Entry(train_frame, textvariable=self.train_size).grid(column=1, row=2)
        
        self.iteration_size = tk.IntVar(value=100000)
        ttk.Label(train_frame, text="Iteration Count").grid(column=0, row=3)
        ttk.Entry(train_frame, textvariable=self.iteration_size).grid(column=1, row=3)

        ttk.Button(train_frame, text="Create Model", command=self.createModel).grid(column=0, row=4)
        
        value_frame = ttk.Labelframe(self.root, text="Create Values")
        value_frame.grid(column=1, row=0)

        self.days = [tk.IntVar(value="") for _ in range(7)] # type: ignore
        self.day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        [
            [
                ttk.Label(value_frame, text=self.day_names[i]).grid(column=0, row=i),
                ttk.Entry(value_frame, textvariable=j).grid(column=1, row=i)
            ] for i, j in enumerate(self.days)
        ]
        
        ttk.Button(value_frame, text="Create Values", command=self.createValues).grid(column=2, row=0)
        ttk.Button(value_frame, text="Show Values", command=self.showValues).grid(column=2, row=1)
        ttk.Button(value_frame, text="Graph Values", command=self.graphValues).grid(column=2, row=2)

        test_frame = ttk.Labelframe(self.root, text="Test")
        test_frame.grid(column=0, row=1, columnspan=2)
        
        test_file_path = tk.StringVar(value="")
        ttk.Label(test_frame, text="Test File Path").grid(column=0, row=0)
        ttk.Entry(test_frame, textvariable=test_file_path).grid(column=1, row=0)
        ttk.Button(test_frame, text="Read Test Data", command=lambda: self.readTestData(test_file_path)).grid(column=2, row=0)

        ttk.Button(test_frame, text="Test Model", command=self.testModel).grid(column=0, row=1)
        ttk.Button(test_frame, text="Show Test", command=self.showTest).grid(column=1, row=1)
        ttk.Button(test_frame, text="Graph Test", command=self.graphTest).grid(column=2, row=1)

        test_metrics = ["NMSE", "RMSE", "MAE", "MAPE", "SMAPE"]
        self.test_metrics_vars = [tk.Variable(), tk.Variable(), tk.Variable(), tk.Variable(), tk.Variable(), tk.Variable()]
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

    def selectColumn(self):
        a = self.input_list.get(self.input_list.curselection())
        self.col_var.set("Selected " + a)
        self.col = a
    
    def showValues(self):
        top = tk.Toplevel(self.root)
        df = pd.DataFrame({j: self.pred[i*24:(i+1)*24] for i, j in enumerate(self.day_names)}) # type: ignore
        pt = Table(top, dataframe=df, editable=False)
        pt.show()

    def graphValues(self):
        plt.plot(self.pred)
        plt.legend(["pred"], loc="upper left")
        plt.show()

    def showTest(self):
        top = tk.Toplevel(self.root)
        df = pd.DataFrame({"Test": self.y_test, "Predict": self.pred}) # type: ignore
        pt = Table(top, dataframe=df, editable=False)
        pt.show()

    def graphTest(self):
        plt.plot(self.y_test) 
        plt.plot(self.pred)
        plt.legend(["test", "pred"], loc="upper left")
        plt.show()

    def createModel(self):
        period = 168
        freq_period = 24
        train = self.df[self.col].iloc[:-self.train_size.get()*period].copy() # type: ignore
        test = self.df[self.col].iloc[-self.train_size.get()*period:].copy() # type: ignore

        for i in range(len(train)//24):
            s = np.sum(train[i*24:(i+1)*24]) / 100
            train.iloc[i*24:(i+1)*24] = np.round(train.iloc[i*24:(i+1)*24]/s, 3)
        
        arr = np.array([train.iloc[i::period].values for i in range(period)])
        arrs = np.array([arr[i*freq_period:(i+1)*freq_period] for i in range(7)])

        us, stds, randoms = [], [], []
        for i in arrs:
            u, s, r = [], [], []
            for j in i:
                u.append(np.mean(j))
                s.append(np.std(j))
                r.append(norm.ppf(np.random.rand(1, self.iteration_size.get()))) # type: ignore
            us.append(u)
            stds.append(s)
            randoms.append(r)

        us = np.array(us)
        stds = np.array(stds)
        randoms = np.array(randoms)

        results = []
        for i in range(us.shape[0]):
            res = []
            for j in range(us.shape[1]):
                res.append(np.clip(us[i][j] + stds[i][j] + randoms[i][j], 0, np.inf))
            results.append(res)
        results = np.array(results).squeeze(axis=2)

        all_maes = []

        for i in range(self.train_size.get()):
            best_maes = []
            to_test = test.iloc[i*168:(i+1)*168]
            for j in range(7):
                idx = to_test.iloc[j*24:(j+1)*24]
                s = idx.sum()
                z = np.broadcast_arrays(idx.values, results[j].T)[0]
                best = np.argmin(mae(z.T, np.round(results[j] * np.sum(z[0]) / 100), multioutput="raw_values"))
                best_maes.append(best)
            all_maes.append(best_maes)

        best_freqs = np.array([[results[i,:,j] for i, j in enumerate(k)] for k in all_maes])
        self.freqs = best_freqs.mean(axis=0)

    def createValues(self):
        l = []
        for i, j in enumerate(self.days):
            l.extend(np.round(self.freqs[i] * j.get() / 100))
        self.pred = np.array(l)

    def testModel(self):
        self.y_test = self.test_df[self.col].iloc[:168].values # type: ignore
        losses = loss(self.y_test, self.pred)
        for i in range(6):
            self.test_metrics_vars[i].set(losses[i])
