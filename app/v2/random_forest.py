import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from pandastable import Table

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split, cross_validate

import os
import json
from pickle import dump as pickle_dump
from pickle import load as pickle_load

from .helpers import loss, skloss, popupmsg

from .components import DatasetInputComponent, ModelValidationComponent, CustomizeTrainsetComponent, TestModelComponent

class RandomForest:
    def __init__(self):
        self.root = ttk.Frame()
        
        # Get Train Set
        self.get_train_set_component = DatasetInputComponent(root_frame=self.root, column=0, row=0)
        
        # Model testing and validation
        self.model_validation_component = ModelValidationComponent(root_frame=self.root, column=0, row=1)

        # Customize Train Set
        self.customize_train_set_component = CustomizeTrainsetComponent(root_frame=self.root, column=0, row=2)
        self.customize_train_set_component.attach_lookback()

        # Model
        model_frame = ttk.Labelframe(self.root, text="Model Frame")
        model_frame.grid(column=1, row=0)

        ## Parameter Optimization
        parameter_optimization_frame = ttk.Labelframe(model_frame, text="Parameter Optimization")
        parameter_optimization_frame.grid(column=0, row=2)

        self.grid_option_var = tk.IntVar(value=0)
        tk.Checkbutton(parameter_optimization_frame, text="Do grid search for optimal parameters", offvalue=0, onvalue=1, variable=self.grid_option_var, command=self.openEntries).grid(column=0, row=0, columnspan=3)

        self.interval_var = tk.IntVar(value=3)
        ttk.Label(parameter_optimization_frame, text="Interval:").grid(column=0, row=1)
        self.interval_entry = ttk.Entry(parameter_optimization_frame, textvariable=self.interval_var, width=8, state=tk.DISABLED)
        self.interval_entry.grid(column=1, row=1, pady=2)

        self.gs_cross_val_option = tk.IntVar(value=0)
        self.gs_cross_val_var = tk.IntVar(value=5)
        tk.Checkbutton(parameter_optimization_frame, text="Cross validate; folds:", offvalue=0, onvalue=1, variable=self.gs_cross_val_option, command=self.openEntries).grid(column=0, row=2)
        self.gs_cross_val_entry = tk.Entry(parameter_optimization_frame, textvariable=self.gs_cross_val_var, state=tk.DISABLED, width=8)
        self.gs_cross_val_entry.grid(column=1, row=2)

        ## Model Parameters
        model_parameters_frame = ttk.LabelFrame(model_frame, text="Model Parameters")
        model_parameters_frame.grid(column=1, row=0, rowspan=3, columnspan=2)
        
        parameter_names = ["N Estimators", "Max Depth", "Min Samples Split", "Min Samples Leaf"]
        self.parameters = [tk.IntVar(value=100), tk.Variable(value="None"), tk.IntVar(value=2), tk.IntVar(value=1)]
        self.optimization_parameters = [[tk.IntVar(value=75), tk.IntVar(value=150)], [tk.IntVar(value=5), tk.IntVar(value=15)], [tk.IntVar(value=2), tk.IntVar(value=4)], [tk.IntVar(value=1), tk.IntVar(value=4)]]
        
        ttk.Label(model_parameters_frame, text="Current").grid(column=1, row=0)
        ttk.Label(model_parameters_frame, text="----- Search Range -----").grid(column=2, row=0, columnspan=2)

        self.model_parameters_frame_options = [
            [
                ttk.Label(model_parameters_frame, text=j+":").grid(column=0, row=i+1),
                ttk.Entry(model_parameters_frame, textvariable=self.parameters[i], state=tk.DISABLED, width=9),
                ttk.Entry(model_parameters_frame, textvariable=self.optimization_parameters[i][0], state=tk.DISABLED, width=9),
                ttk.Entry(model_parameters_frame, textvariable=self.optimization_parameters[i][1], state=tk.DISABLED, width=9)
            ] for i,j in enumerate(parameter_names)
        ]

        for i, j in enumerate(self.model_parameters_frame_options):
            j[1].grid(column=1, row=i+1, padx=2, pady=2, sticky=tk.W)
            j[2].grid(column=2, row=i+1, padx=2, pady=2)
            j[3].grid(column=3, row=i+1, padx=2, pady=2)

        ttk.Button(model_frame, text="Create Model", command=self.createModel).grid(column=0, row=3)
        ttk.Button(model_frame, text="Save Model", command=self.saveModel).grid(column=1, row=3)
        ttk.Button(model_frame, text="Load Model", command=self.loadModel).grid(column=2, row=3)

        # Test Model
        self.test_model_component = TestModelComponent(
            root_frame=self.root, column=1, row=1, 
            forecast_function = self.forecast,
            graph_predicts_function = self.vsGraph,
            show_predict_values_function = self.showPredicts,
        )

        self.openEntries()

    def showPredicts(self):
        print("showp")
        try:
            df = pd.DataFrame({"Test": self.y_test, "Predict": self.pred})
        except:
            return
        top = tk.Toplevel(self.root)
        pt = Table(top, dataframe=df, editable=False)
        pt.show()

    def saveModel(self):
        path = filedialog.asksaveasfilename()
        if not path:
            return
        try:
            self.model: RandomForestRegressor
            model_params = self.model.get_params()
        except:
            popupmsg("Model is not created")
            return

        param_list = []
        param_list.append(list(model_params.items()))
        param_list.append(list(self.get_train_set_component.get_params().items()))
        param_list.append(list(self.model_validation_component.get_params().items()))
        param_list.append(list(self.customize_train_set_component.get_params().items()))

        params = dict(param_list)
        params["is_round"] = self.is_round
        params["is_negative"] = self.is_negative

        os.mkdir(path)
        dump(self.model, path+"/model.joblib")
        if params.get("scale_var") != "None":
            with open(path+"/feature_scaler.pkl", "wb") as f:
                pickle_dump(self.feature_scaler, f)
            with open(path+"/label_scaler.pkl", "wb") as f:
                pickle_dump(self.label_scaler, f)
        if params.get("lookback_option") == 1:
            with open(path+"/last_values.npy", 'wb') as outfile:
                np.save(outfile, self.last)
        if params.get("seasonal_lookback_option") == 1:
            with open(path+"/seasonal_last_values.npy", 'wb') as outfile:
                np.save(outfile, self.seasonal_last)
        with open(path+"/model.json", 'w') as outfile:
            json.dump(params, outfile)

    def loadModel(self):
        path = filedialog.askdirectory()
        if not path:
            return
        try:
            model_path = path + "/model.joblib"
        except:
            popupmsg("There is no model file at the path")
            return

        self.model = load(model_path)
        infile = open(path+"/model.json")
        params = json.load(infile)

        self.predictor_names = params["predictor_names"]
        self.label_name = params["label_name"]
        self.model_validation_component.set_params(params)
        self.customize_train_set_component.set_params(params)

        self.is_round = params.get("is_round", True)
        self.is_negative = params.get("is_negative", False)


        if params["lookback_option"] == 1:
            last_values = open(path+"/last_values.npy", 'rb')
            self.last = np.load(last_values)
            last_values.close()
        try:
            if params["seasonal_lookback_option"] == 1:
                seasonal_last_values = open(path+"/seasonal_last_values.npy", 'rb')
                self.seasonal_last = np.load(seasonal_last_values)
                seasonal_last_values.close()
        except:
            pass

        if params["scale_type"] != "None":
            try:
                with open(path+"/feature_scaler.pkl", "rb") as f:
                    self.feature_scaler = pickle_load(f)
                with open(path+"/label_scaler.pkl", "rb") as f:
                    self.label_scaler = pickle_load(f)
            except:
                pass
        self.parameters[0].set(params["n_estimators"])
        self.parameters[1].set(params["max_depth"])
        self.parameters[2].set(params["min_samples_split"])
        self.parameters[3].set(params["min_samples_leaf"])
       
        self.openEntries()
        msg = f"Predictor names are {self.predictor_names}\nLabel name is {self.label_name}"
        popupmsg(msg)

    def openEntries(self):
        to_open = []
        for i in self.model_parameters_frame_options:
            i[1]["state"] = tk.DISABLED
            i[2]["state"] = tk.DISABLED
            i[3]["state"] = tk.DISABLED

        self.interval_entry["state"] = tk.DISABLED
        self.gs_cross_val_entry["state"] = tk.DISABLED

        if self.grid_option_var.get() and self.gs_cross_val_option.get():
            self.gs_cross_val_entry["state"] = tk.NORMAL

        to_open = list(range(4))
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
    
    def check_errors(self):
        def raise_error(err: bool, msg: str):
            return popupmsg(msg) if err else None

        raise_error(*self.get_train_set_component.check_errors())
        raise_error(*self.model_validation_component.check_errors())
        raise_error(*self.customize_train_set_component.check_errors())
        
        raise_error(*self.test_model_component.check_errors())

        try:
            msg = "Enter a valid Interval for grid search"
            if self.grid_option_var.get() and self.interval_var.get() < 1:
                raise Exception
        
            msg = "Enter a valid Cross Validation fold for grid search (Above 2)"
            if self.gs_cross_val_option.get() and self.gs_cross_val_var.get() < 2:
                raise Exception

        except:
            popupmsg(msg) # type: ignore
            return True

    def getLookback(self, X, y, lookback=0, seasons=0, seasonal_lookback=0, sliding=-1):
        if sliding == 0:
            for i in range(1, lookback+1):
                X[f"t-{i}"] = y.shift(i)
        elif sliding == 1:
            for i in range(1, seasons+1):
                X[f"t-{i*seasonal_lookback}"] = y.shift(i*seasonal_lookback)
        elif sliding == 2:
            for i in range(1, lookback+1):
                X[f"t-{i}"] = y.shift(i)
            for i in range(1, seasons+1):
                X[f"t-{i*seasonal_lookback}"] = y.shift(i*seasonal_lookback)

        X.dropna(inplace=True)
        a = X.to_numpy()
        b = y.iloc[-len(a):].to_numpy().reshape(-1)
        
        if sliding == 0:
            self.last = b[-lookback:]
        elif sliding == 1:
            self.seasonal_last = b[-seasonal_lookback*seasons:]
        elif sliding == 2:
            self.last = b[-(lookback+seasonal_lookback):-seasonal_lookback]
            self.seasonal_last = b[-seasonal_lookback*seasons:]

        return a, b

    def getData(self):
        self.is_round = False
        self.is_negative = False
        lookback_option = self.customize_train_set_component.lookback_option_var.get()
        seasonal_lookback_option = self.customize_train_set_component.seasonal_lookback_option_var.get()
        sliding = lookback_option + 2*seasonal_lookback_option - 1
        self.sliding = sliding
        scale_choice = self.customize_train_set_component.scale_var.get()

        self.predictor_names = self.get_train_set_component.get_predictors()
        self.label_name = self.get_train_set_component.get_label()

        self.df: pd.DataFrame
        X = self.df[self.predictor_names].copy()
        y = self.df[self.label_name].copy()
        
        if y.dtype == int or y.dtype == np.intc or y.dtype == np.int64:
            self.is_round = True
        if any(y < 0):
            self.is_negative = True

        if scale_choice == "StandardScaler":
            self.feature_scaler = StandardScaler()
            self.label_scaler = StandardScaler()
        
            X.iloc[:] = self.feature_scaler.fit_transform(X)
            y.iloc[:] = self.label_scaler.fit_transform(y.values.reshape(-1,1)).reshape(-1)
        
        elif scale_choice == "MinMaxScaler":
            self.feature_scaler = MinMaxScaler()
            self.label_scaler = MinMaxScaler()
            
            X.iloc[:] = self.feature_scaler.fit_transform(X)
            y.iloc[:] = self.label_scaler.fit_transform(y.values.reshape(-1,1)).reshape(-1)
        
        lookback = self.customize_train_set_component.lookback_value_var.get()
        seasonal_period = self.customize_train_set_component.seasonal_period_value_var.get()
        seasonal_lookback = self.customize_train_set_component.seasonal_value_var.get()
            
        X,y = self.getLookback(X, y, lookback, seasonal_period, seasonal_lookback, sliding)

        return X, y


    def createModel(self):
        if self.check_errors():
            return
        
        do_forecast = self.model_validation_component.do_forecast_option_var.get()
        val_option = self.model_validation_component.validation_option_var.get()
        
        X, y = self.getData()
        X: np.ndarray
        y: np.ndarray
            
        scale_type = self.customize_train_set_component.scale_var.get()
        random_percent_value = self.model_validation_component.random_percent_var.get()
        test_metrics_vars = self.test_model_component.test_metrics_vars

        if self.grid_option_var.get() == 0:
            n_estimators = self.parameters[0].get()
            max_depth = self.parameters[1].get()
            max_depth = int(max_depth) if max_depth != "None" else None
            min_samples_split = self.parameters[2].get()
            min_samples_leaf = self.parameters[3].get()

            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
            
            if val_option == 0:
                model.fit(X, y)
                if do_forecast == 0:
                    pred = model.predict(X).reshape(-1)
                    if scale_type != "None":
                        pred = self.label_scaler.inverse_transform(pred.reshape(-1,1)).reshape(-1) # type: ignore
                        y = self.label_scaler.inverse_transform(y.reshape(-1,1)).reshape(-1) # type: ignore
                    losses = loss(y, pred)
                    self.y_test = y
                    self.pred = pred
                    for i,j in enumerate(losses):
                        test_metrics_vars[i].set(j)
                self.model = model # type: ignore
            
            elif val_option == 1:
                if do_forecast == 0:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=random_percent_value/100)
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test).reshape(-1)
                    if scale_type != "None":
                        pred = self.label_scaler.inverse_transform(pred.reshape(-1,1)).reshape(-1) # type: ignore
                        y_test = self.label_scaler.inverse_transform(y_test.reshape(-1,1)).reshape(-1) # type: ignore
                    losses = loss(y_test, pred)
                    self.y_test = y_test
                    self.pred = pred
                    for i,j in enumerate(losses):
                        test_metrics_vars[i].set(j)
                else:
                    size = int((random_percent_value/100)*len(X))
                    X = X[-size:]
                    y = y[-size:]
                    model.fit(X, y)
                self.model = model # type: ignore

            elif val_option == 2:
                if do_forecast == 0:
                    cvs = cross_validate(model, X, y, cv=self.model_validation_component.k_fold_value_var.get(), scoring=skloss)
                    for i,j in enumerate(list(cvs.values())[2:]):
                        self.test_model_component.test_metrics_vars[i].set(j)

            elif val_option == 3:
                if do_forecast == 0:
                    cvs = cross_validate(model, X, y, cv=X.shape[0]-1, scoring=skloss)
                    for i,j in enumerate(list(cvs.values())[2:]):
                        self.test_model_component.test_metrics_vars[i].set(j)
            
        else:
            params = {}
            interval = self.interval_var.get()
             
            params["n_estimators"] = np.unique(np.linspace(self.optimization_parameters[0][0].get(), self.optimization_parameters[0][1].get(), interval, dtype=int))
            params["max_depth"] = np.unique(np.linspace(self.optimization_parameters[1][0].get(), self.optimization_parameters[1][1].get(), interval, dtype=int))
            params["min_samples_split"] = np.unique(np.linspace(self.optimization_parameters[2][0].get(), self.optimization_parameters[2][1].get(), interval, dtype=int))
            params["min_samples_leaf"] = np.unique(np.linspace(self.optimization_parameters[3][0].get(), self.optimization_parameters[3][1].get(), interval, dtype=int))

            cv = self.gs_cross_val_var.get() if self.gs_cross_val_option.get() == 1 else None
            regressor = GridSearchCV(RandomForestRegressor(), params, cv=cv)
            
            if val_option == 0:
                regressor.fit(X, y)
                if do_forecast == 0:
                    pred = regressor.predict(X)
                    if scale_type != "None":
                        pred = self.label_scaler.inverse_transform(pred.reshape(-1,1)).reshape(-1) # type: ignore
                        y = self.label_scaler.inverse_transform(y.reshape(-1,1)).reshape(-1) # type: ignore
                    losses = loss(y, pred)
                    self.y_test = y
                    self.pred = pred
                    for i,j in enumerate(losses):
                        test_metrics_vars[i].set(j)
                self.model = regressor.best_estimator_ # type: ignore

            elif val_option == 1:
                if do_forecast == 0:
                    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=random_percent_value/100)
                    regressor.fit(X_train, y_train)
                    pred = regressor.predict(X_test)
                    if scale_type != "None":
                        pred = self.label_scaler.inverse_transform(pred.reshape(-1,1)).reshape(-1) # type: ignore
                        y_test = self.label_scaler.inverse_transform(y_test.reshape(-1,1)).reshape(-1) # type: ignore
                    losses = loss(y_test, pred)
                    self.y_test = y_test
                    self.pred = pred
                    for i,j in enumerate(losses):
                        test_metrics_vars[i].set(j)
                else:
                    size = int((random_percent_value/100)*len(X))
                    X = X[-size:]
                    y = y[-size:]
                    regressor.fit(X, y)
                self.model = regressor.best_estimator_ # type: ignore
            
            popupmsg("Best Params: " + str(self.model.get_params())) # type: ignore
        
    def forecastLookback(self, num, lookback=0, seasons=0, seasonal_lookback=0, sliding=-1):
        self.test_df: pd.DataFrame
        scale_type = self.customize_train_set_component.scale_var.get()
        pred = []
        if sliding == 0:
            last = self.last
            for i in range(num):
                X_test = self.test_df[self.predictor_names].iloc[i]
                if scale_type != "None":
                    X_test.iloc[:] = self.feature_scaler.transform(X_test.values.reshape(1,-1)).reshape(-1) # type: ignore
                for j in range(1, lookback+1):
                    X_test[f"t-{j}"] = last[-j] # type: ignore
                to_pred = X_test.to_numpy().reshape(1,-1) # type: ignore
                out = self.model.predict(to_pred)
                last = np.append(last, out)[-lookback:]
                pred.append(out)

        elif sliding == 1:
            seasonal_last = self.seasonal_last
            for i in range(num):
                X_test = self.test_df[self.predictor_names].iloc[i]
                if scale_type != "None":
                    X_test.iloc[:] = self.feature_scaler.transform(X_test.values.reshape(1,-1)).reshape(-1) # type: ignore
                for j in range(1, seasons+1):
                    X_test[f"t-{j*seasonal_last}"] = seasonal_last[-j*seasonal_lookback] # type: ignore
                to_pred = X_test.to_numpy().reshape(1,-1) # type: ignore
                out = self.model.predict(to_pred)
                seasonal_last = np.append(seasonal_last, out)[1:]
                pred.append(out)

        elif sliding == 2:
            last = self.last
            seasonal_last = self.seasonal_last
            for i in range(num):
                X_test = self.test_df[self.predictor_names].iloc[i]
                if scale_type != "None":
                    X_test.iloc[:] = self.feature_scaler.transform(X_test.values.reshape(1,-1)).reshape(-1) # type: ignore
                for j in range(1, lookback+1):
                    X_test[f"t-{j}"] = last[-j] # type: ignore
                for j in range(1, seasons+1):
                    X_test[f"t-{j*seasonal_lookback}"] = seasonal_last[-j*seasonal_lookback] # type: ignore
                to_pred = X_test.to_numpy().reshape(1,-1) # type: ignore
                out = self.model.predict(to_pred)
                last = np.append(last, out)[-lookback:]
                seasonal_last = np.append(seasonal_last, out)[1:]
                pred.append(out)

        return np.array(pred).reshape(-1)

    def forecast(self):
        scale_type = self.customize_train_set_component.scale_var.get()
        try:
            num = self.test_model_component.forecast_num_var.get()
        except Exception:
            popupmsg("Enter a valid forecast value")
            return

        lookback_option = self.customize_train_set_component.lookback_option_var.get()
        seasonal_lookback_option = self.customize_train_set_component.seasonal_lookback_option_var.get()
        try:
            X_test = self.test_df[self.predictor_names][:num].to_numpy() # type: ignore
            y_test = self.test_df[self.label_name][:num].to_numpy().reshape(-1) # type: ignore
            self.y_test = y_test
        except Exception:
            popupmsg("Read a test data")
            return
       
        if lookback_option == 0 and seasonal_lookback_option == 0:
            if scale_type != "None":
                X_test = self.feature_scaler.transform(X_test)
            self.pred = self.model.predict(X_test).reshape(-1)
        else:
            sliding = self.sliding
            lookback = self.customize_train_set_component.lookback_value_var.get()
            seasonal_lookback = self.customize_train_set_component.seasonal_value_var.get()
            seasons = self.customize_train_set_component.seasonal_period_value_var.get()
            self.pred = self.forecastLookback(num, lookback, seasons, seasonal_lookback, sliding)

        if scale_type != "None":
            self.pred = self.label_scaler.inverse_transform(self.pred.reshape(-1,1)).reshape(-1) # type: ignore

        if not self.is_negative:
            self.pred = self.pred.clip(0, None)
        if self.is_round:
            self.pred = np.round(self.pred).astype(int)
        
        losses = loss(y_test, self.pred)
        for i, j in enumerate(losses):
            self.test_model_component.test_metrics_vars[i].set(j)

    def vsGraph(self):
        print("vsGraph")
        y_test = self.y_test
        try:
            pred = self.pred
        except:
            return
        plt.plot(y_test)
        plt.plot(pred)
        plt.legend(["test", "pred"], loc="upper left")
        plt.show()
