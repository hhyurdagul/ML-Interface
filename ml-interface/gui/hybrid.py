import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

import numpy as np
import pandas as pd

from pandastable import Table


class Hybrid:

    def __init__(self):
        self.root = ttk.Frame()

        train_frame = ttk.LabelFrame(self.root, text="Train")
        train_frame.grid(column=0, row=0)

        file_path = tk.StringVar(value="")
        ttk.Label(train_frame, text="Train File Path").grid(column=0, row=0)
        ttk.Entry(train_frame, textvariable=file_path).grid(column=1, row=0)
        ttk.Button(train_frame,
                   text="Read Train Data",
                   command=lambda: self.read_csv(file_path)).grid(
                       column=2, row=0)

        self.input_list = tk.Listbox(train_frame)
        self.input_list.grid(column=0, row=1, columnspan=2)
        self.col_var = tk.StringVar(value="Select Column")
        ttk.Button(train_frame,
                   textvariable=self.col_var,
                   command=self.select_column).grid(column=2, row=1)

        self.train_size = tk.IntVar(value=10)
        ttk.Label(train_frame, text="Train Count").grid(column=0, row=2)
        ttk.Entry(train_frame, textvariable=self.train_size).grid(column=1,
                                                                  row=2)

        self.period_size = tk.IntVar(value=48)
        ttk.Label(train_frame, text="Period").grid(column=0, row=3)
        ttk.Entry(train_frame, textvariable=self.period_size).grid(column=1,
                                                                   row=3)

        ttk.Button(train_frame, text="Create Model",
                   command=self.create_model).grid(column=2, row=3)

        test_frame = ttk.LabelFrame(self.root, text="Test")
        test_frame.grid(column=0, row=1)

        test_file_path = tk.StringVar(value="")
        ttk.Label(test_frame, text="Test File Path").grid(column=0, row=0)
        ttk.Entry(test_frame, textvariable=test_file_path).grid(column=1,
                                                                row=0)
        ttk.Button(test_frame,
                   text="Read Test Data",
                   command=lambda: self.read_test_data(test_file_path)).grid(
                       column=2, row=0)

        ttk.Button(test_frame, text="Test Model",
                   command=self.test_model).grid(column=0, row=1)
        ttk.Button(test_frame, text="Show Data",
                   command=self.show_data).grid(column=2, row=1)

    def read_csv(self, file_path):
        path = filedialog.askopenfilename(
            filetypes=[("Csv Files",
                        "*.csv"), ("Xlsx Files",
                                   "*.xlsx"), ("Xlrd Files", ".xls")])
        file_path.set(path)
        if path.endswith(".csv"):
            self.df = pd.read_csv(path)
        else:
            self.df = pd.read_excel(path, index_col=0, engine="openpyxl")

        self.input_list.delete(0, tk.END)
        for i in self.df.columns.tolist():  # type: ignore
            self.input_list.insert(tk.END, i)

    def read_test_data(self, file_path):
        path = filedialog.askopenfilename(
            filetypes=[("Csv Files",
                        "*.csv"), ("Xlsx Files",
                                   "*.xlsx"), ("Xlrd Files", ".xls")])
        file_path.set(path)
        if path.endswith(".csv"):
            self.test_df = pd.read_csv(path)
        else:
            self.test_df = pd.read_excel(path, index_col=0, engine="openpyxl")

    def select_column(self):
        a = self.input_list.get(self.input_list.curselection())
        self.col_var.set("Selected " + a)
        self.col = a

    def show_data(self):
        d = {"Hybrid": self.pred}
        df = pd.DataFrame(d)
        top = tk.Toplevel(self.root)
        pt = Table(top, dataframe=df, editable=False)
        pt.show()

    def create_model(self):
        series = self.df[self.col]
        count = self.train_size.get()
        period = self.period_size.get()

        train = series[:-period]
        val = series[-period:].values

        values = train.iloc[-7 * period * count:].values
        temp = np.array([values[i::period * 7] for i in range(period * 7)])

        u, s = temp.mean(axis=1), temp.std(axis=1)

        up, low = (u + s).round(), (u - s).round()
        self.res = ((up < val) | (low > val)).astype(int)

    def test_model(self):
        series = self.test_df

        pred = []
        for i in range(series.shape[0]):
            if i:
                pred.append(series.iloc[i, 0])
            else:
                pred.append(series.iloc[i, 1])
        self.pred = np.array(pred)
