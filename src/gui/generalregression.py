# Gui
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from pandastable import Table

# Data
import pandas as pd
import numpy as np
from datetime import timedelta

# Sklearn
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split

from pyGRNN import GRNN

from .tools import *

class GeneralizedRegressionNeuralNetwork:
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

        self.date_var = tk.StringVar(value="Select Date Column")
        ttk.Button(get_train_set_frame, textvariable=self.date_var, command=self.selectDate).grid(column=0, row=2)

        ttk.Button(get_train_set_frame, text="Add Predictor", command=self.addPredictor).grid(column=1, row=2)
        ttk.Button(get_train_set_frame, text="Eject Predictor", command=self.ejectPredictor).grid(column=1, row=3)

        ttk.Button(get_train_set_frame, text="Add Target", command=self.addTarget).grid(column=2, row=2)
        ttk.Button(get_train_set_frame, text="Eject Target", command=self.ejectTarget).grid(column=2, row=3)
       
        # Model testing and validation
        model_validation_frame = ttk.Labelframe(self.root, text="Model testing and validation")
        model_validation_frame.grid(column=0, row=1)

        self.validation_option = tk.IntVar(value=0)
        self.random_percent_var = tk.IntVar(value=30)
        self.cross_val_var = tk.IntVar(value=5)
        tk.Radiobutton(model_validation_frame, text="No validation, use all data rows", value=0, variable=self.validation_option).grid(column=0, row=0, columnspan=2, sticky=tk.W)
        tk.Radiobutton(model_validation_frame, text="Random percent", value=1, variable=self.validation_option).grid(column=0, row=1, sticky=tk.W)
        tk.Radiobutton(model_validation_frame, text="K-fold cross-validation", value=2, variable=self.validation_option).grid(column=0, row=2, sticky=tk.W)
        tk.Radiobutton(model_validation_frame, text="Leave one out cross-validation", value=3, variable=self.validation_option).grid(column=0, row=3, columnspan=2, sticky=tk.W)
        ttk.Entry(model_validation_frame, textvariable=self.random_percent_var, width=8).grid(column=1, row=1)
        ttk.Entry(model_validation_frame, textvariable=self.cross_val_var, width=8).grid(column=1, row=2)

    def readCsv(self, file_path):
        path = filedialog.askopenfilename(filetypes=[("Csv Files", "*.csv")])
        file_path.set(path)
        self.df = pd.read_csv(path)
        self.fillInputList()
        
    def fillInputList(self):
        self.input_list.delete(0, tk.END)

        for i in self.df.columns:
            self.input_list.insert(tk.END, i)

    def getTestSet(self, file_path):
        path = filedialog.askopenfilename(filetypes=[("Csv Files", "*.csv")])
        file_path.set(path)
        self.test_df = pd.read_csv(path)

    def showTestSet(self):
        top = tk.Toplevel(self.root)
        df = pd.DataFrame({"Test": self.y_true, "Predict": self.pred})
        pt = Table(top, dataframe=df, editable=False)
        pt.show()

    def selectDate(self):
        a = self.input_list.get(self.input_list.curselection())
        self.date_var.set("Selected " + a)
        date = pd.to_datetime(self.df[a])
        self.df["holiday"] = date.apply(isHoliday)
        self.df["weekend"] = date.apply(isWeekend)
        self.df["hour"] = date.apply(lambda x: x.hour)
        self.df["day"] = date.apply(lambda x: x.day)
        self.df["month"] = date.apply(lambda x: x.month)
        self.df["year"] = date.apply(lambda x: x.year)
        self.fillInputList()
        self.date = date

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
 
