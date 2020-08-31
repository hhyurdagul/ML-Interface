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

from .helpers import *


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

        # Sigma Options
        sigma_options_frame = ttk.LabelFrame(self.root, text="Sigma Options")
        sigma_options_frame.grid(column=1, row=0)

        self.sigma_var = tk.DoubleVar(value=0.4)
        ttk.Label(sigma_options_frame, text="Sigma Value: ").grid(column=0, row=0)
        ttk.Entry(sigma_options_frame, textvariable=self.sigma_var, width=12).grid(column=1, row=0)

        self.find_sigma_option = tk.IntVar(value=0)
        tk.Checkbutton(sigma_options_frame, text="Find best Sigma", offvalue=0, onvalue=1 ,variable=self.find_sigma_option, command=self.openEntries).grid(column=0, row=1, columnspan=2)

        self.minmax_sigma_values = [tk.DoubleVar(value=""), tk.DoubleVar(value=""), tk.IntVar(value="")]
        self.sigma_find_list = [[
            ttk.Label(sigma_options_frame, text=j+"Sigma: ").grid(column=0, row=i+2, pady=5),
            ttk.Entry(sigma_options_frame, textvariable=self.minmax_sigma_values[i], state=tk.DISABLED, width=12)
        ] for i, j in enumerate(["Min. ", "Max. "])]

        [j[1].grid(column=1, row=i+2) for i,j in enumerate(self.sigma_find_list)]

        ttk.Label(sigma_options_frame, text="Search Steps: ").grid(column=0, row=4, pady=5)
        ttk.Entry(sigma_options_frame, textvariable=self.minmax_sigma_values[2], width=12).grid(column=1, row=4)

        self.score = tk.DoubleVar(value="")
        ttk.Button(sigma_options_frame, text="Create Model", command=self.createModel).grid(column=0, row=5, columnspan=2)
        ttk.Label(sigma_options_frame, text="Score: ").grid(column=0, row=6)
        ttk.Entry(sigma_options_frame, textvariable=self.score).grid(column=1, row=6)

        # Test Model
        test_model_frame = ttk.LabelFrame(self.root, text="Test Frame")
        test_model_frame.grid(column=1, row=1)

        ## Test Model Main
        test_model_main_frame = ttk.LabelFrame(test_model_frame, text="Test Model")
        test_model_main_frame.grid(column=0, row=0)

        forecast_num = tk.IntVar(value="")
        ttk.Label(test_model_main_frame, text="# of Forecast").grid(column=0, row=0)
        ttk.Entry(test_model_main_frame, textvariable=forecast_num).grid(column=1, row=0)
        ttk.Button(test_model_main_frame, text="Values", command=self.showTestSet).grid(column=2, row=0)

        test_file_path = tk.StringVar()
        ttk.Label(test_model_main_frame, text="Test File Path").grid(column=0, row=1)
        ttk.Entry(test_model_main_frame, textvariable=test_file_path).grid(column=1, row=1)
        ttk.Button(test_model_main_frame, text="Get Test Set", command=lambda: self.getTestSet(test_file_path)).grid(column=2, row=1)

        ttk.Button(test_model_main_frame, text="Test Model", command=lambda: self.forecast(forecast_num.get())).grid(column=2, row=3)
        ttk.Button(test_model_main_frame, text="Actual vs Forecast Graph", command=self.vsGraph).grid(column=0, row=4, columnspan=3)

        ## Test Model Metrics
        test_model_metrics_frame = ttk.LabelFrame(test_model_frame, text="Test Metrics")
        test_model_metrics_frame.grid(column=1, row=0)

        test_metrics = ["NMSE", "RMSE", "MAE", "MAPE", "SMAPE", "MASE"]
        self.test_metrics_vars = [tk.DoubleVar(), tk.DoubleVar(), tk.DoubleVar(), tk.DoubleVar(), tk.DoubleVar(), tk.DoubleVar()]
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

    def showTestSet(self):
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
    
    def openEntries(self):
        if self.find_sigma_option.get() == 1:
            op = tk.NORMAL
        else:
            op = tk.DISABLED
        for i in self.sigma_find_list:
            i[1]["state"] = op

    def createModel(self):
        op = self.find_sigma_option.get()
        val_option = self.validation_option.get() 
        
        features = self.df[list(self.predictor_list.get(0, tk.END))]
        label = self.df[self.target_list.get(0)]

        if op == 0:
            sigma = self.sigma_var.get()
            model = GRNN(sigma=sigma)

            if val_option == 0:
                model.fit(features, label)
                s = model.score(features, label)
                self.score.set(s)
            elif val_option == 1:
                X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=self.random_percent_var.get())
                model.fit(X_train, y_train)
                s = model.score(X_test, y_test)
                self.score.set(s)
            elif val_option == 2:
                s = cross_val_score(model, features, label, cv=self.cross_val_var.get()).mean()
                self.score.set(s)
                model.fit(features, label)

            self.model = model

        else:
            min_sigma = self.minmax_sigma_values[0].get()
            max_sigma = self.minmax_sigma_values[1].get()
            interval = self.minmax_sigma_values[2].get()

            params = {
                    "sigma": np.unique(np.linspace(min_sigma, max_sigma, interval))
                    }

            reg = GridSearchCV(model, params)
            
            cv = self.cross_val_var.get() if val_option == 2 else 5
            if val_option == 1:
                X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=self.random_percent_var.get())
                reg.fit(X_train, y_train)
                s = reg.fit(X_test, y_test)
                self.score.set(s)
            else: 
                reg.fit(features, label, cv=cv)
                s = reg.score(features, label)
                self.score.set(s)
            
            self.model = reg

    def forecast(self, num):
        X_test = self.test_df[list(self.predictor_list.get(0, tk.END))]
        y_test = self.test_df[self.target_list.get(0)][:num].values.reshape(-1)
        
        self.pred = self.model.predict(X_test)
        self.y_test = y_test
        losses = loss(y_test, self.pred)

        for i in range(6):
            self.test_metrics_vars[i].set(losses[i])

    def vsGraph(self):
        plt.plot(self.pred)
        plt.plot(self.y_test)
        plt.legend(["pred", "test"], loc="upper left")
        plt.show()

