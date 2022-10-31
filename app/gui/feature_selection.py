import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from pandastable import Table

import pandas as pd

from mrmr import mrmr_regression
from sklearn.feature_selection import f_regression, SequentialFeatureSelector
from sklearn.linear_model import LinearRegression

class FS:
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
        
        # Feature Selection
        feature_selection_frame = ttk.Labelframe(self.root, text="Feature Selection")
        feature_selection_frame.grid(column=0, row=1)

        feature_count = tk.IntVar(value=5)
        ttk.Label(feature_selection_frame, text="# Features to select").grid(column=0, row=0)
        ttk.Entry(feature_selection_frame, textvariable=feature_count).grid(column=1, row=0)
        ttk.Button(feature_selection_frame, text="mRMR", command=lambda: self.mrmr_select(feature_count)).grid(column=0, row=1)
        ttk.Button(feature_selection_frame, text="F-Regression", command=lambda: self.f_regression_select(feature_count)).grid(column=1, row=1)
        ttk.Button(feature_selection_frame, text="Forward Sequential Selector", command=lambda: self.forward_sequential_select(feature_count)).grid(column=0, row=2)
        ttk.Button(feature_selection_frame, text="Backward Sequential Selector", command=lambda: self.backward_sequential_select(feature_count)).grid(column=1, row=2)

    def mrmr_select(self, K):
        X = self.df[list(self.predictor_list.get(0, tk.END))]
        y = self.df[self.target_list.get(0)]
        selected_features = mrmr_regression(X=X, y=y, K=K.get())
        top = tk.Toplevel(self.root)
        df = pd.DataFrame(index=range(len(selected_features)), data=selected_features)
        pt = Table(top, dataframe=df, editable=False)
        pt.show()
    
    def f_regression_select(self, K):
        X = self.df[list(self.predictor_list.get(0, tk.END))]
        y = self.df[self.target_list.get(0)]
        selected_features, _ = f_regression(X=X, y=y)
        selected_features = X.columns[selected_features.argsort()[::-1]].values[:K.get()]
        top = tk.Toplevel(self.root)
        df = pd.DataFrame(index=range(len(selected_features)), data=selected_features)
        pt = Table(top, dataframe=df, editable=False)
        pt.show()
    
    def forward_sequential_select(self, K):
        X = self.df[list(self.predictor_list.get(0, tk.END))]
        y = self.df[self.target_list.get(0)]
        selector = SequentialFeatureSelector(estimator=LinearRegression(), n_features_to_select=K.get(), direction="forward")
        selected_indices = selector.fit(X, y).get_support(True)
        selected_features = X.columns[selected_indices]
        top = tk.Toplevel(self.root)
        df = pd.DataFrame(index=range(len(selected_features)), data=selected_features)
        pt = Table(top, dataframe=df, editable=False)
        pt.show()
    
    def backward_sequential_select(self, K):
        X = self.df[list(self.predictor_list.get(0, tk.END))]
        y = self.df[self.target_list.get(0)]
        selector = SequentialFeatureSelector(estimator=LinearRegression(), n_features_to_select=K.get(), direction="backward")
        selected_indices = selector.fit(X, y).get_support(True)
        selected_features = X.columns[selected_indices]
        top = tk.Toplevel(self.root)
        df = pd.DataFrame(index=range(len(selected_features)), data=selected_features)
        pt = Table(top, dataframe=df, editable=False)
        pt.show()
       
    def readCsv(self, file_path):
        path = filedialog.askopenfilename(filetypes=[("Csv Files", "*.csv"), ("Xlsx Files", "*.xlsx"), ("Xlrd Files", ".xls")])
        if not path:
            return
        file_path.set(path)
        if path.endswith(".csv"):
            self.df = pd.read_csv(path) # type: ignore
        else:
            try:
                self.df = pd.read_excel(path)
            except:
                self.df = pd.read_excel(path, engine="openpyxl")
        self.fillInputList()
        
    def fillInputList(self):
        self.input_list.delete(0, tk.END)

        self.df: pd.DataFrame
        for i in self.df.columns.to_list():
            self.input_list.insert(tk.END, i)

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
