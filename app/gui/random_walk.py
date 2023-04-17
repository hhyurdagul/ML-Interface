import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from pandastable import Table
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import os
import json
import random
from .helpers import loss, popupmsg


class RandomWalkRegressor:
    def __init__(self, epsilon, seasonal_value=1):
        self.seed = random.randint(0, 100)
        self.epsilon = epsilon
        self.seasonal_value = seasonal_value

    def get_params(self):
        return {
            "seed": self.seed,
            "epsilon": self.epsilon,
            "seasonal_value": self.seasonal_value,
        }

    def set_series(self, series):
        self.series = series
        self.last = self.series[-self.seasonal_value :].values

    def predict(self, n):
        random.seed(self.seed)
        pred = self.last.tolist()
        for i in range(n):
            direction = random.choice([self.epsilon, -self.epsilon])
            val = pred[i] + direction
            pred.append(val)

        return np.array(pred[self.seasonal_value :])


class RandomWalk:
    def __init__(self):
        self.root = ttk.Frame()

        # Get Train Set
        get_train_set_frame = ttk.Labelframe(self.root, text="Get Train Set")
        get_train_set_frame.grid(column=0, row=0)

        file_path = tk.StringVar(value="")
        ttk.Label(get_train_set_frame, text="Train File Path").grid(column=0, row=0)
        ttk.Entry(get_train_set_frame, textvariable=file_path).grid(column=1, row=0)
        ttk.Button(
            get_train_set_frame,
            text="Read Data",
            command=lambda: self.read_train_data(file_path),
        ).grid(column=2, row=0)

        self.input_list = tk.Listbox(get_train_set_frame)
        self.input_list.grid(column=0, row=1)
        self.input_list.bind("<Double-Button-1>", self.add_predictor)
        self.input_list.bind("<Double-Button-3>", self.add_target)

        self.predictor_list = tk.Listbox(get_train_set_frame)
        self.predictor_list.grid(column=1, row=1)
        self.predictor_list.bind("<Double-Button-1>", self.eject_predictor)

        self.target_list = tk.Listbox(get_train_set_frame)
        self.target_list.grid(column=2, row=1)
        self.target_list.bind("<Double-Button-1>", self.eject_target)

        ttk.Button(
            get_train_set_frame, text="Add Predictor", command=self.add_predictor
        ).grid(column=1, row=2)
        ttk.Button(
            get_train_set_frame, text="Eject Predictor", command=self.eject_predictor
        ).grid(column=1, row=3)

        ttk.Button(get_train_set_frame, text="Add Target", command=self.add_target).grid(
            column=2, row=2
        )
        ttk.Button(
            get_train_set_frame, text="Eject Target", command=self.eject_target
        ).grid(column=2, row=3)

        # Graphs
        graph_frame = ttk.Labelframe(self.root, text="Graphs")
        graph_frame.grid(column=0, row=1)

        self.train_size = tk.IntVar(value=100)
        ttk.Label(graph_frame, text="Train Size").grid(column=0, row=0)
        ttk.Entry(graph_frame, textvariable=self.train_size).grid(column=1, row=0)

        self.train_choice = tk.IntVar(value=0)
        tk.Radiobutton(
            graph_frame, text="As Percent", variable=self.train_choice, value=0
        ).grid(column=0, row=1)
        tk.Radiobutton(
            graph_frame, text="As Number", variable=self.train_choice, value=1
        ).grid(column=1, row=1)

        lags = tk.IntVar(value=40)
        ttk.Label(graph_frame, text="Lag Number").grid(column=0, row=2)
        ttk.Entry(graph_frame, textvariable=lags).grid(column=1, row=2)

        ttk.Button(
            graph_frame, text="Show ACF", command=lambda: self.show_acf(lags.get())
        ).grid(column=0, row=3)

        # Crete Model
        create_model_frame = ttk.Labelframe(self.root, text="Create Model")
        create_model_frame.grid(column=1, row=0)

        self.epsilon_var = tk.DoubleVar(value=10)
        ttk.Label(create_model_frame, text="Epsilon Value: ").grid(column=0, row=0)
        ttk.Entry(create_model_frame, textvariable=self.epsilon_var, width=12).grid(
            column=1, row=0
        )

        self.seasonal_option = tk.IntVar(value=0)
        tk.Checkbutton(
            create_model_frame,
            text="Seasonal",
            offvalue=0,
            onvalue=1,
            variable=self.seasonal_option,
            command=self.open_entries,
        ).grid(column=0, row=1, columnspan=2)
        self.seasonal_value = tk.IntVar(value=12)
        ttk.Label(create_model_frame, text="Seasonal Value", width=12).grid(
            column=0, row=2
        )
        self.seasonal_value_entry = ttk.Entry(
            create_model_frame,
            textvariable=self.seasonal_value,
            width=12,
            state=tk.DISABLED,
        )
        self.seasonal_value_entry.grid(column=1, row=2)

        ttk.Button(
            create_model_frame, text="Create Model", command=self.create_model
        ).grid(column=0, row=5)
        ttk.Button(create_model_frame, text="Save Model", command=self.save_model).grid(
            column=2, row=5
        )

        # Test Model
        test_model_frame = ttk.LabelFrame(self.root, text="Test Frame")
        test_model_frame.grid(column=1, row=1)

        ## Test Model Main
        test_model_main_frame = ttk.LabelFrame(test_model_frame, text="Test Model")
        test_model_main_frame.grid(column=0, row=0)

        self.forecast_num = tk.IntVar(value="")
        ttk.Label(test_model_main_frame, text="# of Forecast").grid(column=0, row=0)
        ttk.Entry(test_model_main_frame, textvariable=self.forecast_num).grid(
            column=1, row=0
        )
        ttk.Button(
            test_model_main_frame, text="Values", command=self.show_predicts
        ).grid(column=2, row=0)

        test_file_path = tk.StringVar()
        ttk.Label(test_model_main_frame, text="Test File Path").grid(column=0, row=1)
        ttk.Entry(test_model_main_frame, textvariable=test_file_path).grid(
            column=1, row=1
        )
        ttk.Button(
            test_model_main_frame,
            text="Get Test Set",
            command=lambda: self.get_test_data(test_file_path),
        ).grid(column=2, row=1)

        ttk.Button(
            test_model_main_frame, text="Load Model", command=self.load_model
        ).grid(column=0, row=3)
        ttk.Button(
            test_model_main_frame,
            text="Forecast",
            command=lambda: self.forecast(self.forecast_num.get()),
        ).grid(column=2, row=3)
        ttk.Button(
            test_model_main_frame, text="Actual vs Forecast Graph", command=self.plot_graph
        ).grid(column=0, row=4, columnspan=3)

        ## Test Model Metrics
        self.test_data_valid = False
        self.forecast_done = False
        test_model_metrics_frame = ttk.LabelFrame(test_model_frame, text="Test Metrics")
        test_model_metrics_frame.grid(column=1, row=0)

        test_metrics = ["NMSE", "RMSE", "MAE", "MAPE", "SMAPE"]
        self.test_metrics_vars = [tk.Variable() for _ in range(len(test_metrics))]
        for i, j in enumerate(test_metrics):
            ttk.Label(test_model_metrics_frame, text=j).grid(column=0, row=i)
            ttk.Entry(
                test_model_metrics_frame, textvariable=self.test_metrics_vars[i]
            ).grid(column=1, row=i)

    def read_train_data(self, file_path):
        path = filedialog.askopenfilename(
            filetypes=[
                ("Csv Files", "*.csv"),
                ("Xlsx Files", "*.xlsx"),
                ("Xlrd Files", ".xls"),
            ]
        )
        file_path.set(path)
        if path.endswith(".csv"):
            self.df = pd.read_csv(path)
        else:
            try:
                self.df = pd.read_excel(path)
            except Exception:
                self.df = pd.read_excel(path, engine="openpyxl")
        self.fill_input_list()

    def fill_input_list(self):
        self.input_list.delete(0, tk.END)

        for i in self.df.columns:
            self.input_list.insert(tk.END, i)

    def get_test_data(self, file_path):
        path = filedialog.askopenfilename(
            filetypes=[
                ("Csv Files", "*.csv"),
                ("Xlsx Files", "*.xlsx"),
                ("Xlrd Files", ".xls"),
            ]
        )
        file_path.set(path)
        if path.endswith(".csv"):
            self.test_df = pd.read_csv(path)
        else:
            try:
                self.test_df = pd.read_excel(path)
            except Exception:
                self.test_df = pd.read_excel(path, engine="openpyxl")

        self.test_data_valid = True
        if self.forecast_done:
            self.forecast(self.forecast_num.get())

    def show_predicts(self):
        d = {}
        if self.test_data_valid:
            self.y_test: np.ndarray
            d["Test"] = self.y_test
        self.pred: np.ndarray
        try:
            d["Predict"] = self.pred
        except Exception:
            return
        df = pd.DataFrame(d)
        top = tk.Toplevel(self.root)
        pt = Table(top, dataframe=df, editable=False)
        pt.show()

    def add_predictor(self, _=None):
        try:
            a = self.input_list.get(self.input_list.curselection())
            if a not in self.predictor_list.get(0, tk.END):
                self.predictor_list.insert(tk.END, a)
        except Exception:
            pass

    def eject_predictor(self, _=None):
        try:
            self.predictor_list.delete(self.predictor_list.curselection())
        except Exception:
            pass

    def add_target(self, _=None):
        try:
            a = self.input_list.get(self.input_list.curselection())
            if self.target_list.size() < 1:
                self.target_list.insert(tk.END, a)
        except Exception:
            pass

    def eject_target(self, _=None):
        try:
            self.target_list.delete(self.target_list.curselection())
        except Exception:
            pass

    def save_model(self):
        path = filedialog.asksaveasfilename()
        if not path:
            return
        try:
            params = self.model.get_params()
        except Exception:
            popupmsg("Model is not created")
            return

        params["predictor_names"] = self.predictor_names
        params["label_name"] = self.label_name
        params["is_round"] = self.is_round
        params["is_negative"] = self.is_negative
        params["seasonal_option"] = self.seasonal_option.get()

        os.mkdir(path)
        with open(path + "/pred.npy", "wb") as outfile:
            np.save(outfile, self.pred)
        with open(path + "/model.json", "w") as outfile:
            json.dump(params, outfile)
        with open(path + "/last.npy", "wb") as outfile:
            np.save(outfile, self.model.last)

    def load_model(self):
        path = filedialog.askdirectory()
        if not path:
            return
        try:
            infile = open(path + "/model.json")
            with open(path + "/pred.npy", "rb") as f:
                self.pred = np.load(f)
        except Exception:
            popupmsg("There is no model file at the path")
            return

        self.model_loaded = True

        params = json.load(infile)
        self.predictor_names = params["predictor_names"]
        self.label_name = params["label_name"]
        self.is_round = params["is_round"]
        self.is_round = True
        self.is_negative = params["is_negative"]

        self.seasonal_option.set(params["seasonal_option"])
        self.seasonal_value.set(params["seasonal_value"])
        self.epsilon_var.set(params["epsilon"])

        self.open_entries()
        names = "\n".join(self.predictor_names)
        msg = f"Predictor names are {names}\nLabel name is {self.label_name}"
        popupmsg(msg)

    def open_entries(self):
        if self.seasonal_option.get() == 1:
            op = tk.NORMAL
        else:
            op = tk.DISABLED
        self.seasonal_value_entry["state"] = op

    def show_acf(self, lags):
        top = tk.Toplevel()
        fig = plt.Figure((20, 15))

        data = self.df[self.target_list.get(0)]
        size = (
            int(self.train_size.get())
            if self.train_choice.get() == 1
            else int((self.train_size.get() / 100) * len(data))
        )
        data = data.iloc[-size:]

        ax = fig.add_subplot(211)
        ax1 = fig.add_subplot(212)

        plot_acf(data, ax=ax, lags=lags)
        plot_pacf(data, ax=ax1, lags=lags)

        canvas = FigureCanvasTkAgg(fig, top)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, top)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def create_model(self):
        self.is_round = False
        self.is_negative = False
        self.model_loaded = False
        self.predictor_names = self.target_list.get(0)
        self.label_name = self.target_list.get(0)
        data = self.df[self.label_name]
        size = (
            int(self.train_size.get())
            if self.train_choice.get() == 1
            else int((self.train_size.get() / 100) * len(data))
        )
        series = data.iloc[-size:]

        if series.dtype == int or series.dtype == np.intc or series.dtype == np.int64:
            self.is_round = True
        if any(series < 0):
            self.is_negative = True

        seasonal_value = self.seasonal_value.get() if self.seasonal_option.get() else 1
        self.model = RandomWalkRegressor(self.epsilon_var.get(), seasonal_value)
        self.model.set_series(series)

    def forecast(self, num):
        if not self.model_loaded:
            print("asd")
            self.pred = self.model.predict(num)

            if not self.is_negative:
                self.pred = self.pred.clip(0, None)
            if self.is_round:
                self.pred = np.round(self.pred).astype(int)

        self.forecast_done = True

        if self.test_data_valid:
            y_test = self.test_df[self.label_name][:num]
            self.y_test = y_test
            losses = loss(y_test, self.pred)

            for i in range(len(self.test_metrics_vars)):
                self.test_metrics_vars[i].set(losses[i])

    def plot_graph(self):
        if self.test_data_valid:
            plt.plot(self.y_test, label="Test")
        try:
            plt.plot(self.pred, label="Predict")
        except Exception:
            return
        plt.legend(loc="upper left")
        plt.show()
