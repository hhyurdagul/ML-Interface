import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from pandastable import Table
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, cross_validate

from datetime import timedelta
import os
import json

# Keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from kerastuner.tuners import RandomSearch

from .helpers import *

class MultiLayerPerceptron:
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
        self.random_percent_var = tk.IntVar(value=70)
        self.cross_val_var = tk.IntVar(value=5)
        tk.Radiobutton(model_validation_frame, text="No validation, use all data rows", value=0, variable=self.validation_option).grid(column=0, row=0, columnspan=2, sticky=tk.W)
        tk.Radiobutton(model_validation_frame, text="Random percent", value=1, variable=self.validation_option).grid(column=0, row=1, sticky=tk.W)
        tk.Radiobutton(model_validation_frame, text="K-fold cross-validation", value=2, variable=self.validation_option).grid(column=0, row=2, sticky=tk.W)
        tk.Radiobutton(model_validation_frame, text="Leave one out cross-validation", value=3, variable=self.validation_option).grid(column=0, row=3, columnspan=2, sticky=tk.W)
        ttk.Entry(model_validation_frame, textvariable=self.random_percent_var, width=8).grid(column=1, row=1)
        ttk.Entry(model_validation_frame, textvariable=self.cross_val_var, width=8).grid(column=1, row=2)
        self.do_forecast_option = tk.IntVar(value=0)
        tk.Checkbutton(model_validation_frame, text="Do Forecast", offvalue=0, onvalue=1, variable=self.do_forecast_option).grid(column=0, row=4, columnspan=2)

        # Create Model
        create_model_frame = ttk.Labelframe(self.root, text="Create Model")
        create_model_frame.grid(column=1, row=0)

        self.do_optimization = False

        ## Model Without Optimization
        model_without_optimization_frame = ttk.Labelframe(create_model_frame, text="Model Without Optimization")
        model_without_optimization_frame.grid(column=0, row=0)

        ttk.Label(model_without_optimization_frame, text="Number of Hidden Layer").grid(column=0, row=0)

        no_optimization_names = ["Neurons in First Layer", "Neurons in Second Layer", "Neurons in Third Layer"]
        self.neuron_numbers_var = [tk.IntVar(value="") for i in range(3)]
        self.activation_var = [tk.StringVar(value="relu") for i in range(3)]
        self.no_optimization_choice_var = tk.IntVar(value=0)

        self.no_optimization = [
                [
                    tk.Radiobutton(model_without_optimization_frame, text=i+1, value=i+1, variable=self.no_optimization_choice_var, command=lambda: self.openLayers(True)).grid(column=i+1, row=0),
                    ttk.Label(model_without_optimization_frame, text=no_optimization_names[i]).grid(column=0, row=i+1),
                    ttk.Entry(model_without_optimization_frame, textvariable=self.neuron_numbers_var[i], state=tk.DISABLED),
                    ttk.Label(model_without_optimization_frame, text="Optimization Function").grid(column=2, row=i+1),
                    ttk.OptionMenu(model_without_optimization_frame, self.activation_var[i], "relu", "relu", "tanh", "sigmoid").grid(column=3, row=i+1)
                ] for i in range(3)
        ]

        for i,j in enumerate(self.no_optimization):
            j[2].grid(column=1, row=i+1)

        ## Model With Optimization
        model_with_optimization_frame = ttk.Labelframe(create_model_frame, text="Model With Optimization")
        model_with_optimization_frame.grid(column=0, row=1)

        optimization_names = ["One Hidden Layer", "Two Hidden Layer", "Three Hidden Layer"]
        self.optimization_choice_var = tk.IntVar(value=0)
        self.min_max_neuron_numbers = [[tk.IntVar(value=""), tk.IntVar(value="")] for i in range(3)]

        self.optimization = [
                [
                    tk.Radiobutton(model_with_optimization_frame, text=optimization_names[i], value=i+1, variable=self.optimization_choice_var, command=lambda: self.openLayers(False)).grid(column=i*2+1, row=0),
                    ttk.Label(model_with_optimization_frame, text=f"N{i+1}_Min").grid(column=i*2, row=1),
                    ttk.Label(model_with_optimization_frame, text=f"N{i+1}_Max").grid(column=i*2, row=2),
                    ttk.Entry(model_with_optimization_frame, textvariable=self.min_max_neuron_numbers[i][0], state=tk.DISABLED),
                    ttk.Entry(model_with_optimization_frame, textvariable=self.min_max_neuron_numbers[i][1], state=tk.DISABLED),
                ] for i in range(3)
        ]

        for i,j in enumerate(self.optimization):
            j[3].grid(column=i*2+1, row=1)
            j[4].grid(column=i*2+1, row=2)


        # Hyperparameters
        hyperparameter_frame = ttk.Labelframe(self.root, text="Hyperparameters")
        hyperparameter_frame.grid(column=1, row=1)

        self.hyperparameters = [tk.IntVar(), tk.IntVar(), tk.StringVar(), tk.StringVar(), tk.DoubleVar(value=0.001), tk.DoubleVar(value=0.0)]

        ttk.Label(hyperparameter_frame, text="Epoch").grid(column=0, row=0)
        ttk.Entry(hyperparameter_frame, textvariable=self.hyperparameters[0]).grid(column=1, row=0)

        ttk.Label(hyperparameter_frame, text="Batch Size").grid(column=2, row=0)
        ttk.Entry(hyperparameter_frame, textvariable=self.hyperparameters[1]).grid(column=3, row=0)

        ttk.Label(hyperparameter_frame, text="Optimizer").grid(column=0, row=1)
        ttk.OptionMenu(hyperparameter_frame, self.hyperparameters[2], "Adam", "Adam", "SGD", "RMSprop").grid(column=1, row=1)

        ttk.Label(hyperparameter_frame, text="Loss_Function").grid(column=2, row=1)
        ttk.OptionMenu(hyperparameter_frame, self.hyperparameters[3], "mean_squared_error", "mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error").grid(column=3, row=1)
        
        ttk.Label(hyperparameter_frame, text="Learning Rate").grid(column=0, row=2)
        ttk.Entry(hyperparameter_frame, textvariable=self.hyperparameters[4]).grid(column=1, row=2)

        ttk.Label(hyperparameter_frame, text="Momentum").grid(column=2, row=2)
        ttk.Entry(hyperparameter_frame, textvariable=self.hyperparameters[5]).grid(column=3, row=2)
        
        self.train_loss = tk.Variable(value="")
        ttk.Button(hyperparameter_frame, text="Create Model", command=self.createModel).grid(column=0, row=5)
        ttk.Label(hyperparameter_frame, text="Train Loss").grid(column=1, row=5)
        ttk.Entry(hyperparameter_frame, textvariable=self.train_loss).grid(column=2, row=5)
        ttk.Button(hyperparameter_frame, text="Save Model", command=self.saveModel).grid(column=3, row=5)

        # Test Model
        test_model_frame = ttk.LabelFrame(self.root, text="Test Frame")
        test_model_frame.grid(column=0, row=2, columnspan=2)

        ## Test Model Main
        test_model_main_frame = ttk.LabelFrame(test_model_frame, text="Test Model")
        test_model_main_frame.grid(column=0, row=0)

        forecast_num = tk.IntVar(value="")
        ttk.Label(test_model_main_frame, text="# of Forecast").grid(column=0, row=0)
        ttk.Entry(test_model_main_frame, textvariable=forecast_num).grid(column=1, row=0)
        ttk.Button(test_model_main_frame, text="Values", command=self.showPredicts).grid(column=2, row=0)

        test_file_path = tk.StringVar()
        ttk.Label(test_model_main_frame, text="Test File Path").grid(column=0, row=1)
        ttk.Entry(test_model_main_frame, textvariable=test_file_path).grid(column=1, row=1)
        ttk.Button(test_model_main_frame, text="Get Test Set", command=lambda: self.getTestSet(test_file_path)).grid(column=2, row=1)

        ttk.Button(test_model_main_frame, text="Test Model", command=lambda: self.forecast(forecast_num.get())).grid(column=2, row=3)
        ttk.Button(test_model_main_frame, text="Actual vs Forecast Graph", command=self.vsGraph).grid(column=0, row=4, columnspan=3)

        ttk.Button(test_model_main_frame, text="Load Model", command=self.loadModel).grid(column=0, row=3)
        
        ## Test Model Metrics
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

    def showPredicts(self):
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

    def openLayers(self, var):
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

    def saveModel(self):
        path = filedialog.asksaveasfilename()
        os.mkdir(path)
        self.model.save(path+"/model.h5")
        params = {
                "predictors": list(self.predictor_list.get(0, tk.END)),
                "validation_option": self.validation_option.get(),
                "do_forecast": self.do_forecast_option.get(),
                "num_layers": self.no_optimization_choice_var.get(),
                "num_neurons": [self.neuron_numbers_var[i].get() for i in range(self.no_optimization_choice_var.get())],
                "hyperparameters": [i.get() for i in self.hyperparameters],
                }
        
        if self.validation_option.get() == 1:
            params["random_percent"] = self.random_percent_var.get()
        
        elif self.validation_option.get() == 2:
            params["k_fold_cv"] = self.cross_val_var.get()

        with open(path+"/model.json", 'w') as outfile:
            json.dump(params, outfile)

    def loadModel(self):
        path = filedialog.askdirectory()
        self.model = load_model(path+"/model.h5")
        infile = open(path+"/model.json")
        params = json.load(infile)
        infile.close()

        self.validation_option.set(params["validation_option"])
        self.do_forecast_option.set(params["do_forecast"])
        self.no_optimization_choice_var.set(params["num_layers"])
        for i in range(params["num_layers"]):
            self.neuron_numbers_var[i].set(params["num_neurons"][i])
        for i, j in enumerate(self.hyperparameters):
            j.set(params["hyperparameters"][i])
    
    def createModel(self):
        X = self.df[list(self.predictor_list.get(0, tk.END))].to_numpy()
        y = self.df[self.target_list.get(0)].to_numpy().reshape(-1)

        layers = self.no_optimization_choice_var.get()
        
        learning_rate = self.hyperparameters[4].get()
        momentum = self.hyperparameters[5].get()

        optimizers = {
                "Adam": Adam(learning_rate=learning_rate),
                "SGD": SGD(learning_rate=learning_rate, momentum=momentum),
                "RMSprop": RMSprop(learning_rate=learning_rate, momentum=momentum)
                }
        
        def base_model():
            model = Sequential()

            for i in range(layers):
                neuron_number = self.neuron_numbers_var[i].get()
                activation = self.activation_var[i].get()
                if i == 0:
                    model.add(Dense(neuron_number, activation=activation, input_dim=X.shape[1]))
                else:
                    model.add(Dense(neuron_number, activation=activation))

            model.add(Dense(1, activation="relu"))
            model.compile(optimizer=optimizers[self.hyperparameters[2].get()], loss=self.hyperparameters[3].get())
            return model

        do_forecast = self.do_forecast_option.get()
        val_option = self.validation_option.get()

        if val_option == 0 or val_option == 1:
            model = base_model()
        elif val_option == 2 or val_option == 3:
            model = KerasRegressor(build_fn=base_model, epochs=self.hyperparameters[0].get(), batch_size=self.hyperparameters[1].get())

        if val_option == 0:
            model.fit(X, y, epochs=self.hyperparameters[0].get(), batch_size=self.hyperparameters[1].get())
            if do_forecast == 0:
                pred = model.predict(X).reshape(-1)
                losses = loss(y, pred)[:-1]
                self.y_test = y
                self.pred = pred
                for i,j in enumerate(losses):
                    self.test_metrics_vars[i].set(j)
            self.model = model

        elif val_option == 1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self.random_percent_var.get()/100)
            model.fit(X_train, y_train, epochs=self.hyperparameters[0].get(), batch_size=self.hyperparameters[1].get())
            if do_forecast == 0:
                pred = model.predict(X_test).reshape(-1)
                losses = loss(y_test, pred)[:-1]
                self.y_test = y_test.reshape(-1)
                self.pred = pred
                for i,j in enumerate(losses):
                    self.test_metrics_vars[i].set(j) 
            self.model = model

        elif val_option == 2:
            cvs = cross_validate(model, X, y, cv=self.cross_val_var.get(), scoring=skloss)
            for i, j in enumerate(list(cvs.values())[2:]):
                self.test_metrics_vars[i].set(j.mean())
        
        elif val_option == 3:
            cvs = cross_validate(model, X, y, cv=X.shape[0]-1, scoring=skloss)
            for i, j in enumerate(list(cvs.values())[2:]):
                self.test_metrics_vars[i].set(j.mean())


    def forecast(self, num):
        
        X_test = self.test_df[list(self.predictor_list.get(0, tk.END))][:num].to_numpy()
        y_test = self.test_df[self.target_list.get(0)][:num].to_numpy().reshape(-1)
        
        self.pred = self.model.predict(X_test).reshape(-1)
        self.y_test = y_test

        losses = loss(y_test, self.pred)
        for i in range(6):
            self.test_metrics_vars[i].set(losses[i])

    def vsGraph(self):
        y_test = self.y_test
        pred = self.pred
        plt.plot(y_test)
        plt.plot(pred)
        plt.legend(["test", "pred"], loc="upper left")
        plt.show()


