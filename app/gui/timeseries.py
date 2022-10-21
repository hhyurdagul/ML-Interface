# Tkinter
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from pandastable import Table
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Default
import os
from datetime import datetime
import json
from pickle import dump as pickle_dump
from pickle import load as pickle_load

# Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Keras
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Input, Flatten, Dropout, Dense
from tensorflow.keras.layers import SimpleRNN, GRU, LSTM, Bidirectional
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.initializers import GlorotUniform, Orthogonal

# Seed
from random import seed
from numpy.random import seed as np_seed # type: ignore
from tensorflow import random
seed(0)
np_seed(0)
random.set_seed(0)

# Helper
from .helpers import loss, popupmsg, cartesian

class TimeSeries:
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
       
        # Customize Train Set
        customize_train_set_frame = ttk.Labelframe(self.root, text="Customize Train Set")
        customize_train_set_frame.grid(column=0, row=1)

        self.train_size_var = tk.IntVar(value="") # type: ignore
        ttk.Label(customize_train_set_frame, text="# of Rows in Train Set").grid(column=0, row=0, columnspan=2, sticky=tk.W)
        ttk.Entry(customize_train_set_frame, textvariable=self.train_size_var, width=8).grid(column=2, row=0, sticky=tk.E)

        self.size_choice_var = tk.IntVar(value=0)
        tk.Radiobutton(customize_train_set_frame, text="As Percent", value=0, variable=self.size_choice_var).grid(column=0, row=1, sticky=tk.W)
        tk.Radiobutton(customize_train_set_frame, text="As Number", value=1, variable=self.size_choice_var).grid(column=1, row=1, sticky=tk.W)

        self.scale_var = tk.StringVar(value="None")
        ttk.Label(customize_train_set_frame, text="Scale Type").grid(column=0, row=2, sticky=tk.W)
        ttk.OptionMenu(customize_train_set_frame, self.scale_var, "None", "None","StandardScaler", "MinMaxScaler").grid(column=1, row=2, columnspan=2)

        self.difference_choice_var = tk.IntVar(value=0)
        self.interval_var = tk.IntVar(value="") # type: ignore
        tk.Checkbutton(customize_train_set_frame, text='Difference Interval', variable=self.difference_choice_var, offvalue=0, onvalue=1, command=self.openDifference).grid(column=0, row=3, columnspan=2, sticky=tk.W)
        self.interval_entry = ttk.Entry(customize_train_set_frame, textvariable=self.interval_var, state=tk.DISABLED, width=8)
        self.interval_entry.grid(column=2, row=3, sticky=tk.E)
 
        self.s_difference_choice_var = tk.IntVar(value=0)
        self.s_interval_var = tk.IntVar(value="") # type: ignore
        tk.Checkbutton(customize_train_set_frame, text='Second Difference Interval', variable=self.s_difference_choice_var, offvalue=0, onvalue=1, command=self.openDifference).grid(column=0, row=4, columnspan=2, sticky=tk.W)
        self.s_interval_entry = ttk.Entry(customize_train_set_frame, textvariable=self.s_interval_var, state=tk.DISABLED, width=8)
        self.s_interval_entry.grid(column=2, row=4, sticky=tk.E)

        # Lag Options
        lag_options_frame = ttk.Labelframe(self.root, text="Lag Options")
        lag_options_frame.grid(column=0, row=2)

        self.acf_lags = tk.IntVar(value=40)
        ttk.Label(lag_options_frame, text="Upper Lag Limit").grid(column=0, row=0, sticky=tk.W)
        ttk.Entry(lag_options_frame, textvariable=self.acf_lags, width=8).grid(column=1, row=0, sticky=tk.W)
        ttk.Button(lag_options_frame, text="Show ACF", command=lambda: self.showACF(self.acf_lags.get())).grid(column=2, row=0, sticky=tk.W)

        self.lag_option_var = tk.IntVar(value="") # type: ignore
        tk.Radiobutton(lag_options_frame, text="Use All Lags", value=0, variable=self.lag_option_var, command=self.openEntries).grid(column=0, row=1, sticky=tk.W)
        tk.Radiobutton(lag_options_frame, text="Use Selected(1,3,..)", value=1, variable=self.lag_option_var, command=self.openEntries).grid(column=0, row=2, sticky=tk.W)
        tk.Radiobutton(lag_options_frame, text="Use Best N", value=2, variable=self.lag_option_var, command=self.openEntries).grid(column=0, row=3, sticky=tk.W)
        tk.Radiobutton(lag_options_frame, text="Use Correlation > n (Between 0-1)", value=3, variable=self.lag_option_var, command=self.openEntries).grid(column=0, row=4, sticky=tk.W)
        
        self.lag_entries = [ttk.Entry(lag_options_frame, state=tk.DISABLED) for _ in range(4)]
        [self.lag_entries[i-1].grid(column=1, row=i, columnspan=2) for i in range(1,5)]

        # Create Model
        create_model_frame = ttk.Labelframe(self.root, text="Create Model")
        create_model_frame.grid(column=1, row=0)
        
        self.model_instance = 0
        self.runtime = datetime.now().strftime("%d/%m/%Y %H:%M")
        self.do_optimization = False
        
        ## Model Without Optimization
        model_without_optimization_frame = ttk.Labelframe(create_model_frame, text="Model Without Optimization")
        model_without_optimization_frame.grid(column=0, row=0)

        ttk.Label(model_without_optimization_frame, text="Number of Hidden Layer").grid(column=0, row=0)
        
        layer_count = 20
        self.layer_count = layer_count

        self.neuron_numbers_var = [tk.IntVar(value="") for _ in range(layer_count)] # type: ignore
        self.activation_var = [tk.StringVar(value="relu") for _ in range(layer_count)]
        self.no_optimization_choice_var = tk.IntVar(value=0)
        
        self.no_optimization = [
                [
                    tk.Radiobutton(model_without_optimization_frame, text=str(i+1), value=i+1, variable=self.no_optimization_choice_var, command=lambda: self.openOptimizationLayers(True)).grid(column=i+1, row=0),
                    ttk.Label(model_without_optimization_frame, text=f"Neurons in {i+1}. Layer:").grid(column=0, row=i+1),
                    ttk.Entry(model_without_optimization_frame, textvariable=self.neuron_numbers_var[i], state=tk.DISABLED),
                    ttk.Label(model_without_optimization_frame, text="Activation Function").grid(column=3, row=i+1, columnspan=2),
                    ttk.OptionMenu(model_without_optimization_frame, self.activation_var[i], "relu", "relu", "tanh", "sigmoid", "linear").grid(column=5, row=i+1)
                ] for i in range(5)
        ]

        self.output_activation = tk.StringVar(value="relu")
        ttk.Label(model_without_optimization_frame, text="Output Activation").grid(column=1, row=7),
        ttk.OptionMenu(model_without_optimization_frame, self.output_activation, "relu", "relu", "tanh", "sigmoid", "linear").grid(column=2, row=7)

        top_level = tk.Toplevel(self.root)
        top_level.protocol("WM_DELETE_WINDOW", top_level.withdraw)
        top = ttk.Frame(top_level)
        top.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        for i in range(5, 20):
            self.no_optimization.append(
                    [
                        tk.Radiobutton(top, text=str(i+1), value=i+1, variable=self.no_optimization_choice_var, command=lambda: self.openOptimizationLayers(True)).grid(column=i-5, row=0),
                        ttk.Label(top, text=f"Neurons in {i+1}. Layer:").grid(column=0, row=i+1-5, columnspan=4),
                        ttk.Entry(top, textvariable=self.neuron_numbers_var[i], state=tk.DISABLED),
                        ttk.Label(top, text="Activation Function").grid(column=9, row=i+1-5, columnspan=4),
                        ttk.OptionMenu(top, self.activation_var[i], "relu", "relu", "tanh", "sigmoid", "linear").grid(column=13, row=i+1-5, columnspan=3)
                    ]
            )
        
        for i,j in enumerate(self.no_optimization):
            if i > 4:
                j[2].grid(column=4, row=i+1-5, columnspan=5)
            else:
                j[2].grid(column=1, row=i+1, columnspan=2)
        
        top_level.withdraw()
        
        ttk.Button(model_without_optimization_frame, text="More Layers", command=top_level.deiconify).grid(column=3, row=7)
        ttk.Button(top, text="Done", command=top_level.withdraw).grid(column=0, row=16)

        ## Model With Optimization
        model_with_optimization_frame = ttk.Labelframe(create_model_frame, text="Model With Optimization")
        model_with_optimization_frame.grid(column=0, row=1)

        optimization_names = {1:"One Hidden Layer", 2:"Two Hidden Layer", 3:"Three Hidden Layer"}
        self.optimization_choice_var = tk.IntVar(value=0)

        self.neuron_min_number_var = [tk.IntVar(value="") for _ in range(3)] # type: ignore
        self.neuron_max_number_var = [tk.IntVar(value="") for _ in range(3)] # type: ignore

        self.optimization = [
                [
                    tk.Radiobutton(model_with_optimization_frame, text=optimization_names[i+1], value=i+1, variable=self.optimization_choice_var, command=lambda: self.openOptimizationLayers(False)).grid(column=i*2+1, row=0),
                    ttk.Label(model_with_optimization_frame, text=f"N{i+1}_Min").grid(column=i*2, row=1),
                    ttk.Label(model_with_optimization_frame, text=f"N{i+1}_Max").grid(column=i*2, row=2),
                    ttk.Entry(model_with_optimization_frame, textvariable=self.neuron_min_number_var[i], state=tk.DISABLED),
                    ttk.Entry(model_with_optimization_frame, textvariable=self.neuron_max_number_var[i], state=tk.DISABLED)
                ] for i in range(3)
        ]
        
        for i,j in enumerate(self.optimization):
            j[3].grid(column=i*2+1, row=1)
            j[4].grid(column=i*2+1, row=2)

        
        # Hyperparameters
        hyperparameter_frame = ttk.Labelframe(self.root, text="Hyperparameters")
        hyperparameter_frame.grid(column=1, row=1)

        self.hyperparameters = {"Epoch": tk.IntVar(), "Batch_Size": tk.IntVar(), "Optimizer": tk.StringVar(), "Loss_Function": tk.StringVar(), "Learning_Rate": tk.Variable(value=0.001), "Momentum": tk.Variable(value=0.0)}
        
        ttk.Label(hyperparameter_frame, text="Epoch").grid(column=0, row=0)
        ttk.Entry(hyperparameter_frame, textvariable=self.hyperparameters["Epoch"]).grid(column=1, row=0)

        ttk.Label(hyperparameter_frame, text="Batch Size").grid(column=2, row=0)
        ttk.Entry(hyperparameter_frame, textvariable=self.hyperparameters["Batch_Size"]).grid(column=3, row=0)

        ttk.Label(hyperparameter_frame, text="Optimizer").grid(column=0, row=1)
        ttk.OptionMenu(hyperparameter_frame, self.hyperparameters["Optimizer"], "Adam", "Adam", "SGD", "RMSprop").grid(column=1, row=1)

        ttk.Label(hyperparameter_frame, text="Loss_Function").grid(column=2, row=1)
        ttk.OptionMenu(hyperparameter_frame, self.hyperparameters["Loss_Function"], "mean_squared_error", "mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error").grid(column=3, row=1)

        ttk.Label(hyperparameter_frame, text="Learning Rate").grid(column=0, row=2)
        ttk.Entry(hyperparameter_frame, textvariable=self.hyperparameters["Learning_Rate"]).grid(column=1, row=2)

        ttk.Label(hyperparameter_frame, text="Momentum (Between 0-1)").grid(column=2, row=2)
        ttk.Entry(hyperparameter_frame, textvariable=self.hyperparameters["Momentum"]).grid(column=3, row=2)

        model_names = ["MLP Model", "CNN Model", "LSTM Model", "Bi-LSTM Model"]
        second_model_names = ["RNN Model", "GRU Model", "CNN-LSTM Model"]
        self.model_var = tk.IntVar(value=0)
        ttk.Label(hyperparameter_frame, text="Model Type").grid(column=0, row=3, columnspan=4)
        [tk.Radiobutton(hyperparameter_frame, text=model_names[i], value=i, variable=self.model_var).grid(column=i, row=4) for i in range(4)]
        [tk.Radiobutton(hyperparameter_frame, text=second_model_names[i], value=i+4, variable=self.model_var).grid(column=i, row=5) for i in range(3)]

        self.train_loss = tk.Variable(value="")
        ttk.Button(hyperparameter_frame, text="Create Model", command=self.createModel).grid(column=0, row=6)
        ttk.Label(hyperparameter_frame, text="Train Loss").grid(column=1, row=6)
        ttk.Entry(hyperparameter_frame, textvariable=self.train_loss).grid(column=2, row=6)
        ttk.Button(hyperparameter_frame, text="Save Model", command=self.saveModel).grid(column=3, row=6)

        ttk.Label(hyperparameter_frame, text="Best Model Neuron Numbers").grid(column=0, row=7)
        self.best_model_neurons = [tk.IntVar(value="") for _ in range(3)] # type: ignore
        [ttk.Entry(hyperparameter_frame, textvariable=self.best_model_neurons[i], width=5).grid(column=i+1, row=7) for i in range(3)]
       
        # Test Model
        test_model_frame = ttk.Labelframe(self.root, text="Test Frame")
        test_model_frame.grid(column=1, row=2)
      
        ## Test Model Main
        test_model_main_frame = ttk.LabelFrame(test_model_frame, text="Test Model")
        test_model_main_frame.grid(column=0, row=0)

        self.forecast_num = tk.IntVar(value="") # type: ignore
        ttk.Label(test_model_main_frame, text="# of Forecast").grid(column=0, row=0)
        ttk.Entry(test_model_main_frame, textvariable=self.forecast_num).grid(column=1, row=0)
        ttk.Button(test_model_main_frame, text="Values", command=self.showTestSet).grid(column=2, row=0)

        test_file_path = tk.StringVar()
        ttk.Label(test_model_main_frame, text="Test File Path").grid(column=0, row=1)
        ttk.Entry(test_model_main_frame, textvariable=test_file_path).grid(column=1, row=1)
        ttk.Button(test_model_main_frame, text="Get Test Set", command=lambda: self.getTestSet(test_file_path)).grid(column=2, row=1)

        ttk.Button(test_model_main_frame, text="Forecast", command=self.testModel).grid(column=0, row=2)
        ttk.Button(test_model_main_frame, text="Actual vs Forecasted Graph", command=self.vsGraph).grid(column=1, row=2)
        ttk.Button(test_model_main_frame, text="Load Model", command=self.loadModel).grid(column=0, row=4)

        ## Test Model Metrics
        self.test_data_valid = False
        self.forecast_done = False
        test_model_metrics_frame = ttk.LabelFrame(test_model_frame, text="Test Metrics")
        test_model_metrics_frame.grid(column=1, row=0)

        test_metrics = ["NMSE", "RMSE", "MAE", "MAPE", "SMAPE"]
        self.test_metrics_vars = [tk.Variable() for _ in range(len(test_metrics))]
        for i, j in enumerate(test_metrics):
            ttk.Label(test_model_metrics_frame, text=j).grid(column=0, row=i)
            ttk.Entry(test_model_metrics_frame, textvariable=self.test_metrics_vars[i], width=8).grid(column=1,row=i, padx=3)

    def readCsv(self, file_path):
        path = filedialog.askopenfilename(filetypes=[("Csv Files", "*.csv"), ("Excel Files", "*.xl*")])
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

        self.input_list.delete(0, tk.END)
        self.predictor_list.delete(0, tk.END)
        self.target_list.delete(0, tk.END)

        for i in self.df.columns: # type: ignore
            self.input_list.insert(tk.END, i)

    def getTestSet(self, file_path):
        path = filedialog.askopenfilename(filetypes=[("Csv Files", "*.csv"), ("Excel Files", "*.xl*")])
        if not path:
            return
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
            self.testModel()

    def showTestSet(self):
        d = {}
        if self.test_data_valid:
            self.y_test: np.ndarray
            d["Test"] = self.y_test[:,0]
        self.pred: np.ndarray
        try:
            d["Predict"] = self.pred[:,0]
        except:
            return
        df = pd.DataFrame(d)
        top = tk.Toplevel(self.root)
        pt = Table(top, dataframe=df, editable=False)
        pt.show()

    def saveModel(self):
        path = filedialog.asksaveasfilename()
        if not path:
            return
        try:
            params = {
                "predictor_names": self.predictor_names,
                "label_name": self.label_name,
                "is_round": self.is_round,
                "is_negative": self.is_negative,
                "train_size": self.train_size_var.get(),
                "size_choice": self.size_choice_var.get(),
                "scale_type": self.scale_var.get(),
                "difference_choice": self.difference_choice_var.get(),
                "interval": self.interval_var.get() if self.difference_choice_var.get() else None,
                "second_difference_choice": self.s_difference_choice_var.get(),
                "second_interval": self.s_interval_var.get() if self.s_difference_choice_var.get() else None,
                "acf_lags": self.acf_lags.get(),
                "lag_choice": self.lag_option_var.get(),
                "lag_number": self.lag_entries[self.lag_option_var.get()].get(),
                "num_layers": self.no_optimization_choice_var.get(),
                "num_neurons": [self.neuron_numbers_var[i].get() for i in range(self.no_optimization_choice_var.get())],
                "activations": [self.activation_var[i].get() for i in range(self.no_optimization_choice_var.get())],
                "output_activation": self.output_activation.get(),
                "hyperparameters": {i:j.get() for (i, j) in self.hyperparameters.items()},
                "model": self.model_var.get(),
                "train_loss": self.train_loss.get()
                }
        except:
            popupmsg("Model is not created")
            return

        os.mkdir(path)
        self.model.save(path+"/model.h5") # type: ignore

        if self.scale_var.get() != "None":
            with open(path+"/feature_scaler.pkl", "wb") as f:
                pickle_dump(self.feature_scaler, f)
            with open(path+"/label_scaler.pkl", "wb") as f:
                pickle_dump(self.label_scaler, f)

        if self.difference_choice_var.get():
            with open(path+"/fill.npy", "wb") as outfile:
                np.save(outfile, self.fill_values)
        if self.s_difference_choice_var.get():
            with open(path+"/s_fill.npy", "wb") as outfile:
                np.save(outfile, self.s_fill_values) # type: ignore

        with open(path+"/lags.npy", 'wb') as outfile:
            np.save(outfile, self.lags)
        with open(path+"/last_values.npy", 'wb') as outfile:
            np.save(outfile, self.last)

        with open(path+"/model.json", 'w') as outfile:
            json.dump(params, outfile)

    def loadModel(self):
        path = filedialog.askdirectory()
        if not path:
            return
        try:
            self.model = load_model(path+"/model.h5")
        except:
            popupmsg("There is no model file at the path")
        infile = open(path+"/model.json")
        params = json.load(infile)
        infile.close()
        last_values = open(path+"/last_values.npy", 'rb')
        self.last = np.load(last_values)
        last_values.close()
        lags = open(path+"/lags.npy", 'rb')
        self.lags = np.load(lags)
        lags.close()
        
        self.predictor_names = params["predictor_names"]
        self.label_name = params["label_name"]
        try:
            self.is_round = params["is_round"]
        except:
            self.is_round = True
        try:
            self.is_negative = False
        except:
            self.is_negative = params["is_negative"]
        self.train_size_var.set(params["train_size"])
        self.size_choice_var.set(params["size_choice"])
        self.scale_var.set(params["scale_type"])
        if params["scale_type"] != "None":
            try:
                with open(path+"/feature_scaler.pkl", "rb") as f:
                    self.feature_scaler = pickle_load(f)
                with open(path+"/label_scaler.pkl", "rb") as f:
                    self.label_scaler = pickle_load(f)
            except:
                pass
        self.difference_choice_var.set(params["difference_choice"])
        if params["difference_choice"] == 1:
            self.interval_var.set(params["interval"])
            try:
                with open(path+"/fill.npy", "rb") as f:
                    self.fill_values = np.load(f)
            except:
                pass
        self.s_difference_choice_var.set(params["second_difference_choice"])
        if params["second_difference_choice"] == 1:
            self.s_interval_var.set(params["second_interval"])
            try:
                with open(path+"/s_fill.npy", "rb") as f:
                    self.s_fill_values = np.load(f)
            except:
                pass
        self.acf_lags.set(params["acf_lags"])
        self.lag_option_var.set(params["lag_choice"])
        self.openEntries()
        self.lag_entries[params["lag_choice"]].delete(0,tk.END)
        self.lag_entries[params["lag_choice"]].insert(0, params["lag_number"])
        self.no_optimization_choice_var.set(params["num_layers"])
        [self.neuron_numbers_var[i].set(j) for i,j in enumerate(params["num_neurons"])]
        [self.activation_var[i].set(j) for i,j in enumerate(params["activations"])]
        try:
            self.output_activation.set(params["output_activation"])
        except:
            self.output_activation.set("relu")
        [self.hyperparameters[i].set(j) for (i,j) in params["hyperparameters"].items()]
        self.model_var.set(params["model"])
        self.train_loss.set(params["train_loss"])
        self.openOptimizationLayers(True)

        msg = f"Predictor names are {self.predictor_names}\nLabel name is {self.label_name}"
        popupmsg(msg)

        #features, label = self.getDataset()
        #self.createLag(features, label)

    def addPredictor(self, _=None):
        try:
            a = self.input_list.get(self.input_list.curselection())
            if self.predictor_list.size() < 1:
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

    def openDifference(self):
        s = tk.NORMAL if self.difference_choice_var.get() else tk.DISABLED
        self.interval_entry["state"] = s
        
        s_s = tk.NORMAL if self.s_difference_choice_var.get() else tk.DISABLED
        self.s_interval_entry["state"] = s_s

    def showACF(self, lags):
        if not self.target_list.get(0):
            popupmsg("Select a target")
            return
        top = tk.Toplevel()
        fig = plt.Figure((10, 8))
        
        self.df: pd.DataFrame
        data = self.df[self.target_list.get(0)]

        ax = fig.add_subplot(211)
        ax1 = fig.add_subplot(212)
        """
        if self.s_difference_choice_var.get() and self.difference_choice_var.get():
            f_diff = self.interval_var.get()
            s_diff = self.s_interval_var.get()
            first_diff: pd.Series
            first_diff = data.diff(f_diff)[f_diff:] # type: ignore
            plot_acf(first_diff.diff(s_diff)[s_diff:], ax=ax, lags=lags)
            plot_pacf(first_diff.diff(s_diff)[s_diff:], ax=ax1, lags=lags)

        elif self.s_difference_choice_var.get():
            f_diff = self.s_interval_var.get()
            plot_acf(data.diff(f_diff)[f_diff:], ax=ax, lags=lags)
            plot_pacf(data.diff(f_diff)[f_diff:], ax=ax1, lags=lags)
        
        elif self.difference_choice_var.get():
            f_diff = self.interval_var.get()
            plot_acf(data.diff(f_diff)[f_diff:], ax=ax, lags=lags)
            plot_pacf(data.diff(f_diff)[f_diff:], ax=ax1, lags=lags)
 
        else:
            plot_acf(data, ax=ax, lags=lags)
            plot_pacf(data, ax=ax1, lags=lags)
        """
        plot_acf(data, ax=ax, lags=lags)
        plot_pacf(data, ax=ax1, lags=lags)
        
        canvas = FigureCanvasTkAgg(fig, top)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, top)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def openEntries(self):
        o = self.lag_option_var.get()
        for i, j in enumerate(self.lag_entries):
            if i == o:
                j["state"] = tk.NORMAL
            else:
                j["state"] = tk.DISABLED

    def openOptimizationLayers(self, var):
        for i in self.no_optimization:
            i[2]["state"] = tk.DISABLED

        for i in self.optimization:
            i[3]["state"] = tk.DISABLED
            i[4]["state"] = tk.DISABLED

        if var:
            for i in range(self.no_optimization_choice_var.get()):
                self.no_optimization[i][2]["state"] = tk.NORMAL
            self.optimization_choice_var.set(0)
            
            self.do_optimization = False

        if not var:
            for i in range(self.optimization_choice_var.get()):
                self.optimization[i][3]["state"] = tk.NORMAL
                self.optimization[i][4]["state"] = tk.NORMAL
            self.no_optimization_choice_var.set(0)
            self.do_optimization = True

    def checkErrors(self):
        try:
            msg = "Read a data first"
            self.df.head(1)
            
            msg = "Select a predictor"
            if not self.predictor_list.get(0):
                raise Exception
            
            msg = "Select a target"
            if not self.target_list.get(0):
                raise Exception
            
            msg = "Enter a valid train size"
            if self.train_size_var.get() <= 0:
                raise Exception
            
            msg = "Enter a valid difference interval"
            if self.difference_choice_var.get():
                self.interval_var.get()
            
            msg = "Enter a valid second difference interval"
            if self.s_difference_choice_var.get():
                self.s_interval_var.get()

            msg = "Enter a lag number bigger than 1"
            if self.acf_lags.get() <= 1:
                raise Exception

            msg = "Select a valid lag choice"
            self.lag_option_var.get()
            
            msg = "Enter a valid lag number"
            if not self.lag_entries[self.lag_option_var.get()].get():
                raise Exception

            msg = "Lag size cannot be bigger than the data size"
            if self.lag_option_var.get() == 0 and int(self.lag_entries[0].get()) > len(self.df):
                raise Exception
            elif self.lag_option_var.get() == 2 and int(self.lag_entries[2].get()) > len(self.df):
                raise Exception
            
            msg = "Acf cannot be bigger than 1 nor less than 0"
            if self.lag_option_var.get() == 3 and (float(self.lag_entries[3].get()) >= 1 or float(self.lag_entries[3].get()) <= 0):
                raise Exception
 
            msg = "Select a valid layer number"
            if not self.no_optimization_choice_var.get():
                raise Exception

            msg = "Enter a valid neuron number"
            neuron_empty = False
            for i in range(self.no_optimization_choice_var.get()):
                try:
                    self.neuron_numbers_var[i].get()
                except:
                    neuron_empty = True
            if neuron_empty:
                raise Exception
            
            msg = "Enter a valid Epoch size"
            if self.hyperparameters["Epoch"].get() <= 0:
                raise Exception

            msg = "Enter a valid Batch size"
            if self.hyperparameters["Batch_Size"].get() <= 0:
                raise Exception

            msg = "Enter a valid Learning Rate"
            if float(self.hyperparameters["Learning_Rate"].get()) <= 0:
                raise Exception
            
            msg = "Enter a valid Momentum value"
            if self.hyperparameters["Optimizer"].get() != "Adam" and float(self.hyperparameters["Momentum"].get()) < 0:
                raise Exception

            return False
            
        except:
            popupmsg(msg) # type: ignore
            return True

    def difference(self, data, diff, interval=None, fill_values=None):
        if diff:
            return np.array([data[i] - data[i-interval] for i in range(interval, len(data))])
        else:
            for i in range(len(data)):
                if i >= interval:
                    data[i] = data[i] + data[i-interval]
                else:
                    data[i] = data[i] + fill_values[(len(fill_values) - interval)+i]

    def getDataset(self):
        self.is_round = False
        self.is_negative = False
        scale_choice = self.scale_var.get()
        difference_choice = self.difference_choice_var.get()
        
        size_choice = self.size_choice_var.get()
        size = self.train_size_var.get() if size_choice==1 else (self.train_size_var.get()/100) * len(self.df)
        size = int(size)

        self.predictor_names = self.predictor_list.get(0)
        self.label_name = self.target_list.get(0)
        features = self.df[[self.predictor_names]].iloc[-size:].copy().to_numpy()
        label = self.df[[self.label_name]].iloc[-size:].copy().to_numpy()
        
        if label.dtype == int or label.dtype == np.intc or label.dtype == np.int64:
            self.is_round = True

        if any(label < 0):
            self.is_negative = True

        if scale_choice == "StandardScaler":
            self.feature_scaler = StandardScaler()
            self.label_scaler = StandardScaler()
            
            features = self.feature_scaler.fit_transform(features)
            label = self.label_scaler.fit_transform(label)
        
        elif scale_choice == "MinMaxScaler":
            self.feature_scaler = MinMaxScaler()
            self.label_scaler = MinMaxScaler()
            
            features = self.feature_scaler.fit_transform(features)
            label = self.label_scaler.fit_transform(label)

        if difference_choice:
            self.fill_values = label
            interval = self.interval_var.get()
            features = self.difference(features, True, interval)
            label = self.difference(label, True, interval)

        if self.s_difference_choice_var.get():
            self.s_fill_values = label
            s_interval = self.s_interval_var.get()
            features = self.difference(features, True, s_interval)
            label = self.difference(label, True, s_interval)

        return features, label

    def getLags(self, features, label, n):
        X, y = [], []
        for i in range(len(features) - n):
            X.append(features[i:i+n])
            y.append(label[i+n])
        
        self.last = np.array(features[len(features)-n:])

        return np.array(X), np.array(y)

    def createLag(self, features, label):
        lag_type = self.lag_option_var.get()
        acf_lags = self.acf_lags.get()
        acf_vals = acf(self.df[self.label_name].values, nlags=acf_lags, fft=False) # type: ignore

        if lag_type == 0:
            max_lag = int(self.lag_entries[0].get())
            self.lags = list(range(max_lag))

        elif lag_type == 1:
            lag = self.lag_entries[1].get()
            self.lags = [int(i) for i in lag.split(',')]
            max_lag = max(self.lags) + 1

        elif lag_type == 2:
            lag = self.lag_entries[2].get()
            numbers = np.argsort(acf_vals[1:])[-int(lag):] # type: ignore
            self.lags = np.sort(numbers)
            max_lag = max(self.lags) + 1

        # lag type == 3
        else:
            lag = self.lag_entries[3].get()
            numbers = np.array(acf_vals[1:]) # type: ignore
            self.lags = np.where(numbers>float(lag))[0]
            max_lag = max(self.lags) + 1

        X, y = self.getLags(features, label, max_lag)
        return X, y 

    def createModel(self):
        self.model_instance += 1
        clear_session()
        if self.checkErrors():
            return

        features, label = self.getDataset()
        X, y = self.createLag(features, label)
        X = X[:, self.lags]

        learning_rate = float(self.hyperparameters["Learning_Rate"].get())
        if self.hyperparameters["Optimizer"] != "Adam":
            momentum = float(self.hyperparameters["Momentum"].get())
        else:
            momentum = 0.0

        optimizers = {
                "Adam": Adam(learning_rate=learning_rate),
                "SGD": SGD(learning_rate=learning_rate, momentum=momentum),
                "RMSprop": RMSprop(learning_rate=learning_rate, momentum=momentum)
                }

        shape = (X.shape[1], X.shape[2])
        model_choice = self.model_var.get()

        if not self.do_optimization:
            X_train = X
            y_train = y
            model = Sequential()
            model.add(Input(shape=shape))
            
            if model_choice == 0:
                model.add(Flatten())

            layers = self.no_optimization_choice_var.get()
            for i in range(layers):
                neuron_number = self.neuron_numbers_var[i].get()
                activation_function = self.activation_var[i].get()
                if model_choice == 0:
                    model.add(Dense(neuron_number, activation=activation_function, kernel_initializer=GlorotUniform(seed=0)))
                    model.add(Dropout(0.2))
                
                elif model_choice == 1:
                    model.add(Conv1D(filters=neuron_number, kernel_size=2, activation=activation_function, kernel_initializer=GlorotUniform(seed=0)))
                    model.add(MaxPooling1D(pool_size=2))
                
                elif model_choice == 2:
                    if i == layers-1:
                        model.add(LSTM(neuron_number, activation=activation_function, return_sequences=False, kernel_initializer=GlorotUniform(seed=0), recurrent_initializer=Orthogonal(seed=0)))
                        model.add(Dropout(0.2))
                    else:
                        model.add(LSTM(neuron_number, activation=activation_function, return_sequences=True, kernel_initializer=GlorotUniform(seed=0), recurrent_initializer=Orthogonal(seed=0)))
                        model.add(Dropout(0.2))

                elif model_choice == 3:
                    if i == layers-1:
                        model.add(Bidirectional(LSTM(neuron_number, activation=activation_function, return_sequences=False, kernel_initializer=GlorotUniform(seed=0), recurrent_initializer=Orthogonal(seed=0))))
                        model.add(Dropout(0.2))
                    else:
                        model.add(Bidirectional(LSTM(neuron_number, activation=activation_function, return_sequences=True, kernel_initializer=GlorotUniform(seed=0), recurrent_initializer=Orthogonal(seed=0))))
                        model.add(Dropout(0.2))

                elif model_choice == 4:
                    if i == layers-1:
                        model.add(SimpleRNN(neuron_number, activation=activation_function, return_sequences=False, kernel_initializer=GlorotUniform(seed=0), recurrent_initializer=Orthogonal(seed=0)))
                        model.add(Dropout(0.2))
                    else:
                        model.add(SimpleRNN(neuron_number, activation=activation_function, return_sequences=True, kernel_initializer=GlorotUniform(seed=0), recurrent_initializer=Orthogonal(seed=0)))
                        model.add(Dropout(0.2))

                #elif model_choice == 5:
                else: 
                    if i == layers-1:
                        model.add(GRU(neuron_number, activation=activation_function, return_sequences=False, kernel_initializer=GlorotUniform(seed=0), recurrent_initializer=Orthogonal(seed=0)))
                        model.add(Dropout(0.2))
                    else:
                        model.add(GRU(neuron_number, activation=activation_function, return_sequences=True, kernel_initializer=GlorotUniform(seed=0), recurrent_initializer=Orthogonal(seed=0)))
                        model.add(Dropout(0.2))
            
            if model_choice == 1:
                model.add(Flatten())
                model.add(Dense(32, kernel_initializer=GlorotUniform(seed=0)))

            model.add(Dense(1, activation=self.output_activation.get(), kernel_initializer=GlorotUniform(seed=0)))
            model.compile(optimizer = optimizers[self.hyperparameters["Optimizer"].get()], loss=self.hyperparameters["Loss_Function"].get())
            
            history = model.fit(X_train, y_train, epochs=self.hyperparameters["Epoch"].get(), batch_size=self.hyperparameters["Batch_Size"].get(), verbose=1, shuffle=False)
            loss = history.history["loss"][-1]
            self.train_loss.set(loss)
            self.model = model

        else:
            X_train, X_test = X[:-60], X[-60:]
            y_train, y_test = y[:-60], y[-60:]
            
            def eval(model):
                model.compile(optimizer = optimizers[self.hyperparameters["Optimizer"].get()], loss=self.hyperparameters["Loss_Function"].get())
                model.fit(X_train, y_train, epochs=self.hyperparameters["Epoch"].get(), batch_size=self.hyperparameters["Batch_Size"].get(), verbose=0, shuffle=False)
                return model.evaluate(X_test, y_test)

            best_score = np.inf
            best_neurons = [0]
            best_model = None
            layers = self.optimization_choice_var.get()
            mins = self.neuron_min_number_var
            maxs = self.neuron_max_number_var
            range1 = np.unique(np.linspace(mins[0].get(), maxs[0].get(), 10, dtype=np.uint16))
            
            if layers == 1:
                clear_session()
                for k, i in enumerate(range1):
                    print(f"{k+1}.Model initialized")
                    if model_choice == 0:
                        model = Sequential([Input(shape=shape), Flatten(), Dense(i, activation="relu"), Dense(1, activation="relu")])
                    elif model_choice == 1:
                        model = Sequential([Input(shape=shape), Conv1D(i, activation="relu"), Flatten(), Dense(1, activation="relu")])
                    elif model_choice == 2:
                        model = Sequential([Input(shape=shape), LSTM(i, activation="relu"), Dense(1, activation="relu")])
                    elif model_choice == 3:
                        model = Sequential([Input(shape=shape), Bidirectional(LSTM(i, activation="relu")), Dense(1, activation="relu")])
                    elif model_choice == 4:
                        model = Sequential([Input(shape=shape), SimpleRNN(i, activation="relu"), Dense(1, activation="relu")])
                    else:
                        model = Sequential([Input(shape=shape), GRU(i, activation="relu"), Dense(1, activation="relu")])

                    score = eval(model)
                    print("Score: "+ str(score))
                    if best_score >= score:
                        best_neurons = [i]
                        best_score = score
                        print("Best Score: "+ str(score))
                        best_model = model
            else:
                clear_session()
                range2 = np.unique(np.linspace(mins[1].get(), maxs[1].get(), 10, dtype=np.uint16))
                if layers == 2:
                    arr = cartesian(range1, range2)
                    for k, i in enumerate(arr):
                        print(f"{k+1}.Model initialized")
                        if model_choice == 0:
                            model = Sequential([Input(shape=shape), Flatten(), Dense(i[0], activation="relu"), Dense(i[1], activation="relu"), Dense(1, activation="relu")])
                        elif model_choice == 1:
                            model = Sequential([Input(shape=shape), Conv1D(i[0], activation="relu"), Conv1D(i[1], activation="relu"), Flatten(), Dense(1, activation="relu")])
                        elif model_choice == 2:
                            model = Sequential([Input(shape=shape), LSTM(i[0], activation="relu", return_sequences=True), LSTM(i[1], activation="relu"), Dense(1, activation="relu")])
                        elif model_choice == 3:
                            model = Sequential([Input(shape=shape), Bidirectional(LSTM(i[0], activation="relu", return_sequences=True)), Bidirectional(LSTM(i[1], activation="relu")), Dense(1, activation="relu")])
                        elif model_choice == 4:
                            model = Sequential([Input(shape=shape), SimpleRNN(i[0], activation="relu", return_sequences=True), SimpleRNN(i[1], activation="relu"), Dense(1, activation="relu")])
                        else:
                            model = Sequential([Input(shape=shape), GRU(i[0], activation="relu", return_sequences=True), GRU(i[1], activation="relu"), Dense(1, activation="relu")])

                        score = eval(model)
                        if best_score >= score:
                            best_neurons = i
                            best_score = score
                            best_model = model
                else: 
                    range3 = np.unique(np.linspace(mins[2].get(), maxs[2].get(), 10, dtype=np.uint16))
                    arr = cartesian(range1, range2, range3)
                    for k, i in enumerate(arr):
                        print(f"{k+1}.Model initialized")
                        if model_choice == 0:
                            model = Sequential([Input(shape=shape), Flatten(), Dense(i[0], activation="relu"), Dense(i[1], activation="relu"), Dense(i[2], activation="relu"), Dense(1, activation="relu")])
                        elif model_choice == 1:
                            model = Sequential([Input(shape=shape), Conv1D(i[0], activation="relu"), Conv1D(i[1], activation="relu"), Conv1D(i[2], activation="relu"), Flatten(), Dense(1, activation="relu")])
                        elif model_choice == 2:
                            model = Sequential([Input(shape=shape), LSTM(i[0], activation="relu", return_sequences=True), LSTM(i[1], activation="relu", return_sequences=True), LSTM(i[2], activation="relu"), Dense(1, activation="relu")])
                        elif model_choice == 3:
                            model = Sequential([Input(shape=shape), Bidirectional(LSTM(i[0], activation="relu", return_sequences=True)), Bidirectional(LSTM(i[1], activation="relu", return_sequences=True)), Bidirectional(LSTM(i[2], activation="relu")), Dense(1, activation="relu")])
                        elif model_choice == 4:
                            model = Sequential([Input(shape=shape), SimpleRNN(i[0], activation="relu", return_sequences=True), SimpleRNN(i[1], activation="relu", return_sequences=True), SimpleRNN(i[2], activation="relu"), Dense(1, activation="relu")])
                        else:
                            model = Sequential([Input(shape=shape), GRU(i[0], activation="relu", return_sequences=True), GRU(i[1], activation="relu", return_sequences=True), GRU(i[2], activation="relu"), Dense(1, activation="relu")])
                        score = eval(model)
                        if best_score >= score:
                            best_neurons = i
                            best_score = score
                            best_model = model

            for i in self.best_model_neurons:
                i.set("") # type: ignore
            for i, j in enumerate(best_neurons):
                self.best_model_neurons[i].set(j)
            
            best_model.compile(optimizer = optimizers[self.hyperparameters["Optimizer"].get()], loss=self.hyperparameters["Loss_Function"].get())
            
            history = best_model.fit(X, y, epochs=self.hyperparameters["Epoch"].get(), batch_size=self.hyperparameters["Batch_Size"].get(), verbose=1, shuffle=False)
            loss = history.history["loss"][-1]
            self.train_loss.set(loss)
            self.model = best_model

    def testModel(self):
        try:
            num = self.forecast_num.get()
        except:
            popupmsg("Enter a valid Forecast value")
            return

        input_value = self.last
        steps, features = input_value.shape[0], input_value.shape[1]
        shape = (1,steps,features)
        pred = []

        for _ in range(num):
            output = self.model.predict(input_value.reshape(shape)[:, self.lags], verbose=0)
            pred = np.append(pred, output)
            input_value = np.append(input_value, output)[-shape[1]:]

        self.pred = np.array(pred).reshape(-1,1)
        
        if self.s_difference_choice_var.get():
            self.difference(self.pred, False, self.s_interval_var.get(), self.s_fill_values)
        
        if self.difference_choice_var.get():
            self.difference(self.pred, False, self.interval_var.get(), self.fill_values)
        
        if self.scale_var.get() != "None":
            self.pred = self.label_scaler.inverse_transform(self.pred)
        
        if not self.is_negative:
            self.pred = self.pred.clip(0, None)
        if self.is_round:
            self.pred = np.round(self.pred).astype(int)
        
        if self.test_data_valid:
            self.y_test = self.test_df[[self.label_name]]
            self.y_test = np.asarray(self.y_test)[:num]
            self.y_test: np.ndarray

            losses = loss(self.y_test, self.pred)
            for i in range(len(self.test_metrics_vars)):
                self.test_metrics_vars[i].set(losses[i])
        self.forecast_done = True

    def vsGraph(self):
        if self.test_data_valid:
            plt.plot(self.y_test, label="Test")
        try:
            plt.plot(self.pred, label="Predict")
        except:
            return
        plt.legend(loc="upper left")
        plt.show()
