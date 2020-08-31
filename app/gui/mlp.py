import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from pandastable import Table
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.svm import SVR, NuSVR

from datetime import timedelta

from .helpers import loss

class SupportVectorMachine:
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


        # Model
        model_frame = ttk.Labelframe(self.root, text="Model Frame")
        model_frame.grid(column=1, row=0)

        ## Model Type
        model_type_frame = ttk.Labelframe(model_frame, text="Type of SVR Model")
        model_type_frame.grid(column=0, row=0)

        self.model_type_var = tk.IntVar(value=0)
        tk.Radiobutton(model_type_frame, text="Epsilon-SVR", value=0, variable=self.model_type_var, command=self.openEntries).grid(column=0, row=0)
        tk.Radiobutton(model_type_frame, text="Nu-SVR", value=1, variable=self.model_type_var, command=self.openEntries).grid(column=1, row=0)

        ## Kernel Type
        kernel_type_frame = ttk.Labelframe(model_frame, text="Kernel Function")
        kernel_type_frame.grid(column=0, row=1)

        self.kernel_type_var = tk.IntVar()
        tk.Radiobutton(kernel_type_frame, text="Linear", value=0, variable=self.kernel_type_var, command=self.openEntries).grid(column=0, row=0, sticky=tk.W)
        tk.Radiobutton(kernel_type_frame, text="RBF", value=1, variable=self.kernel_type_var, command=self.openEntries).grid(column=1, row=0, sticky=tk.W)
        tk.Radiobutton(kernel_type_frame, text="Polynomial", value=2, variable=self.kernel_type_var, command=self.openEntries).grid(column=0, row=1)
        tk.Radiobutton(kernel_type_frame, text="Sigmoid", value=3, variable=self.kernel_type_var, command=self.openEntries).grid(column=1, row=1)
        
        ## Parameter Optimization
        parameter_optimization_frame = ttk.Labelframe(model_frame, text="Parameter Optimization")
        parameter_optimization_frame.grid(column=0, row=2)

        self.grid_option_var = tk.IntVar(value=0)
        tk.Checkbutton(parameter_optimization_frame, text="Do grid search for optimal parameters", offvalue=0, onvalue=1, variable=self.grid_option_var, command=self.openEntries).grid(column=0, row=0, columnspan=3)

        self.interval_var = tk.IntVar(value="")
        ttk.Label(parameter_optimization_frame, text="Interval:").grid(column=0, row=1)
        self.interval_entry = ttk.Entry(parameter_optimization_frame, textvariable=self.interval_var, width=8, state=tk.DISABLED)
        self.interval_entry.grid(column=1, row=1, pady=2)

        self.gs_cross_val_option = tk.IntVar(value=0)
        self.gs_cross_val_var = tk.IntVar(value="")
        tk.Checkbutton(parameter_optimization_frame, text="Cross validate; folds:", offvalue=0, onvalue=1, variable=self.gs_cross_val_option, command=self.openEntries).grid(column=0, row=2)
        self.gs_cross_val_entry = tk.Entry(parameter_optimization_frame, textvariable=self.gs_cross_val_var, state=tk.DISABLED, width=8)
        self.gs_cross_val_entry.grid(column=1, row=2)

        ## Model Parameters
        model_parameters_frame = ttk.LabelFrame(model_frame, text="Model Parameters")
        model_parameters_frame.grid(column=1, row=0, rowspan=3, columnspan=2)
        
        parameter_names = ["Epsilon", "Nu", "C", "Gamma", "Coef0", "Degree"]
        self.parameters = [tk.Variable(value="0.10"), tk.Variable(value="0.50"), tk.Variable(value="1.00"), tk.Variable(value="0.10"), tk.Variable(value="0.00"), tk.Variable(value="3.00")]
        self.optimization_parameters = [[tk.Variable(), tk.Variable()], [tk.Variable(), tk.Variable()], [tk.Variable(), tk.Variable()], [tk.Variable(), tk.Variable()], [tk.Variable(), tk.Variable()], [tk.Variable(), tk.Variable()]]
        
        ttk.Label(model_parameters_frame, text="Current").grid(column=1, row=0)
        ttk.Label(model_parameters_frame, text="------ Search Range ------").grid(column=2, row=0, columnspan=2)

        self.model_parameters_frame_options = [
            [
                ttk.Label(model_parameters_frame, text=j+":").grid(column=0, row=i+1),
                ttk.Entry(model_parameters_frame, textvariable=self.parameters[i], state=tk.DISABLED, width=9),
                ttk.Entry(model_parameters_frame, textvariable=self.optimization_parameters[i][0], state=tk.DISABLED, width=9),
                ttk.Entry(model_parameters_frame, textvariable=self.optimization_parameters[i][1], state=tk.DISABLED, width=9)
            ] for i,j in enumerate(parameter_names)
        ]

        for i, j in enumerate(self.model_parameters_frame_options):
            j[1].grid(column=1, row=i+1, padx=2, pady=2)
            j[2].grid(column=2, row=i+1, padx=2, pady=2)
            j[3].grid(column=3, row=i+1, padx=2, pady=2)

        ttk.Label(model_parameters_frame, text="Gamma:").grid(column=0, row=7)
        self.gamma_choice = tk.IntVar(value=0)
        self.gamma_radios = [tk.Radiobutton(model_parameters_frame, text=j, value=i, variable=self.gamma_choice, state=tk.DISABLED, command=self.openEntries) for i, j in enumerate(["Scale", "Auto", "Value"])]
        
        for i, j in enumerate(self.gamma_radios):
            j.grid(column=i+1, row=7)
    
        ttk.Button(model_frame, text="Create Model", command=self.createModel).grid(column=0, row=3)

        self.svm_train_score = tk.Variable(value="")
        ttk.Label(model_frame, text="SVM Train Score:").grid(column=1, row=3)
        ttk.Entry(model_frame, textvariable=self.svm_train_score, width=12).grid(column=2, row=3)

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
        self.test_metrics_vars = [tk.Variable(), tk.Variable(), tk.Variable(), tk.Variable(), tk.Variable(), tk.Variable()]
        for i, j in enumerate(test_metrics):
            ttk.Label(test_model_metrics_frame, text=j).grid(column=0, row=i)
            ttk.Entry(test_model_metrics_frame, textvariable=self.test_metrics_vars[i]).grid(column=1,row=i)

        self.openEntries()


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
        to_open = []
        for i in self.model_parameters_frame_options:
            i[1]["state"] = tk.DISABLED
            i[2]["state"] = tk.DISABLED
            i[3]["state"] = tk.DISABLED

        self.interval_entry["state"] = tk.DISABLED
        self.gs_cross_val_entry["state"] = tk.DISABLED

        if self.kernel_type_var.get() == 0:
            for i in self.gamma_radios:
                i["state"] = tk.DISABLED
        else:
            for i in self.gamma_radios:
                i["state"] = tk.NORMAL
            if self.gamma_choice.get() == 2:
                to_open.append(3)
        
        if self.gs_cross_val_option.get() == 1:
            self.gs_cross_val_entry["state"] = tk.NORMAL

        if self.kernel_type_var.get() == 2:
            to_open.append(5)
            to_open.append(4)
        elif self.kernel_type_var.get() == 3:
            to_open.append(4) 

        to_open.append(self.model_type_var.get())
        to_open.append(2)

        to_open.sort()
        
        opt = self.grid_option_var.get()

        self.open(to_open, opt)

    def open(self, to_open, opt=0):
        if opt == 1:
            self.interval_entry["state"] = tk.NORMAL
            for i in to_open:
                self.model_parameters_frame_options[i][2]["state"] = tk.NORMAL
                self.model_parameters_frame_options[i][3]["state"] = tk.NORMAL
        else:
            for i in to_open:
                self.model_parameters_frame_options[i][1]["state"] = tk.NORMAL
        
        self.vars_nums = to_open
    
    def createModel(self):
        gamma_choice = self.gamma_choice.get()
        kernels = ["linear", "rbf", "poly", "sigmoid"]
        kernel = kernels[self.kernel_type_var.get()]
        
        df = self.df
        self.features = df[list(self.predictor_list.get(0, tk.END))]
        self.label = df[self.target_list.get(0)]
        
        X = self.features
        y = self.label

        if self.grid_option_var.get() == 0:
            epsilon = float(self.parameters[0].get())
            nu = float(self.parameters[1].get())
            C = float(self.parameters[2].get())
            gamma = float(self.parameters[3].get()) if gamma_choice == 2 else "auto" if gamma_choice == 1 else "scale"
            coef0 = float(self.parameters[4].get())
            degree = float(self.parameters[5].get())
            
            if self.model_type_var.get() == 0:
                model = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma, coef0=coef0, degree=degree)
            else:
                model = NuSVR(kernel=kernel, C=C, nu=nu, gamma=gamma, coef0=coef0, degree=degree)

            if self.validation_option.get() == 0:
                model.fit(X, y)
                s = model.score(X, y)
                self.svm_train_score.set(s)
            
            elif self.validation_option.get() == 1:
                X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=self.random_percent_var.get()/100)
                model.fit(X_train, y_train)
                s = model.score(X_test, y_test)
                self.svm_train_score.set(s)

            elif self.validation_option.get() == 2:
                cvs = cross_val_score(model, X, y, cv=self.cross_val_var.get()).mean()             
                self.svm_train_score.set(cvs)

            elif self.validation_option.get() == 3:
                cvs = cross_val_score(model, X, y, cv=X.shape[0]-1).mean()
                self.svm_train_score.set(cvs)
            
            model.fit(X, y)
            self.model = model
            
        else:
            params = {}
            interval = self.interval_var.get()
            
            params["C"] = np.unique(np.linspace(float(self.optimization_parameters[2][0].get()), float(self.optimization_parameters[2][1].get()), interval))
            if self.model_type_var.get() == 0:
                params["epsilon"] = np.unique(np.linspace(float(self.optimization_parameters[0][0].get()), float(self.optimization_parameters[0][1].get()), interval))
                model = SVR()
            else:
                min_nu = max(0.0001, float(self.optimization_parameters[1][0].get()))
                max_nu = min(1, float(self.optimization_parameters[1][1].get()))
                params["nu"] = np.unique(np.linspace(min_nu, max_nu, interval))
                model = NuSVR()
            if kernel != "linear":
                if gamma_choice == 2:
                    params["gamma"] = np.unique(np.linspace(float(self.optimization_parameters[3][0].get()), float(self.optimization_parameters[3][1].get()), interval))
                elif gamma_choice == 1:
                    params["gamma"] = ["auto"]
                else:
                    params["gamma"] = ["scale"]
            
            if kernel == "poly" or kernel == "sigmoid":
                params["coef0"] = np.unique(np.linspace(float(self.optimization_parameters[4][0].get()), float(self.optimization_parameters[4][1].get()), interval))

            if kernel == "poly":
                params["degree"] = np.unique(np.linspace(float(self.optimization_parameters[5][0].get()), float(self.optimization_parameters[5][1].get()), interval, dtype=int))

            params["kernel"] = [kernel]

            cv = self.gs_cross_val_var.get() if self.gs_cross_val_option.get() == 1 else None
            
            regressor = GridSearchCV(model, params, cv=cv)
            regressor.fit(X, y)
            s = regressor.score(X, y)
            self.svm_train_score.set(s)

            self.model = regressor
        
    def forecast(self, num):
        
        X_test = self.test_df[list(self.predictor_list.get(0, tk.END))][:num].values
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


