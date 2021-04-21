# Gui
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from pandastable import Table

# Data
import pandas as pd

from ReliefF import ReliefF
from pymrmre import mrmr

from .helpers import *


class FeatureSelection:
    def __init__(self):
        self.root = ttk.Frame()

        file_path = tk.StringVar(value="")
        ttk.Label(self.root, text="File Path").grid(column=0, row=0)
        ttk.Entry(self.root, textvariable=file_path).grid(column=1, row=0)
        ttk.Button(self.root, text="Read Dataset", command=lambda: self.readCsv(file_path)).grid(column=2, row=0)

        self.input_list = tk.Listbox(self.root)
        self.input_list.grid(column=0, row=1)
        self.input_list.bind("<Double-Button-1>", self.addPredictor)
        self.input_list.bind("<Double-Button-3>", self.addTarget)

        self.predictor_list = tk.Listbox(self.root)
        self.predictor_list.grid(column=1, row=1)
        self.predictor_list.bind("<Double-Button-1>", self.ejectPredictor)

        self.target_list = tk.Listbox(self.root)
        self.target_list.grid(column=2, row=1)
        self.target_list.bind("<Double-Button-1>", self.ejectTarget)

        ttk.Button(self.root, text="Add Predictor", command=self.addPredictor).grid(column=1, row=2)
        ttk.Button(self.root, text="Eject Predictor", command=self.ejectPredictor).grid(column=1, row=3)

        ttk.Button(self.root, text="Add Target", command=self.addTarget).grid(column=2, row=2)
        ttk.Button(self.root, text="Eject Target", command=self.ejectTarget).grid(column=2, row=3)

        ttk.Button(self.root, text="ReliefF Algoritm", command=lambda: self.featureSelect(False)).grid(column=1, row=4)
        ttk.Button(self.root, text="MRMR Algoritm", command=lambda: self.featureSelect(True)).grid(column=2, row=4)
       
    def readCsv(self, file_path):
        path = filedialog.askopenfilename(filetypes=[("Csv Files", "*.csv"), ("Excel Files", "*.xl*")])
        file_path.set(path)
        if path.endswith(".csv"):
            self.df = pd.read_csv(path)
        else:
            self.df = pd.read_excel(path, engine="openpyxl")
        self.fillInputList()
        
    def fillInputList(self):
        self.input_list.delete(0, tk.END)
        self.predictor_list.delete(0, tk.END)
        self.target_list.delete(0, tk.END)

        for i in self.df.columns:
            self.input_list.insert(tk.END, i)

    def addPredictor(self, event=None):
        try:
            a = self.input_list.get(self.input_list.curselection())
            if a not in self.predictor_list.get(0,tk.END):
                self.predictor_list.insert(tk.END, a)
        except:
            pass

    def ejectPredictor(self, event=None):
        try:
            self.predictor_list.delete(self.predictor_list.curselection())
        except:
            pass
    
    def addTarget(self, event=None):
        try:
            a = self.input_list.get(self.input_list.curselection())
            if self.target_list.size() < 1:
                self.target_list.insert(tk.END, a)
        except:
            pass

    def ejectTarget(self, event=None):
        try:
            self.target_list.delete(self.target_list.curselection())
        except:
            pass

    def featureSelect(self, mode=False):
        feature_names = list(self.predictor_list.get(0, tk.END))
        label_name = [self.target_list.get(0)]
        X, y = self.df[feature_names], self.df[label_name]
        length = len(feature_names)
        if mode:
            sol = mrmr.mrmr_ensemble(features=X.astype("double"), targets=y, solution_length=length)
            l = sol[0][0]
        else:
            rf = ReliefF(n_neighbors=len(y)//5, n_features_to_keep=length)
            f = rf.fit_transform(X.values, y.values.reshape(-1))
            l = X.columns[rf.top_features].tolist()

        top = tk.Toplevel(self.root)
        df = pd.DataFrame({"Features": l})
        pt = Table(top, dataframe=df, editable=False)
        pt.show()
