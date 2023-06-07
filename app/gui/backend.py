import json
import tkinter as tk
from tkinter import ttk


class BaseSupervisedModel:
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

        ttk.Button(
            get_train_set_frame, text="Add Target", command=self.add_target
        ).grid(column=2, row=2)
        ttk.Button(
            get_train_set_frame, text="Eject Target", command=self.eject_target
        ).grid(column=2, row=3)
        
        # Model testing and validation
        model_validation_frame = ttk.Labelframe(
            self.root, text="Model testing and validation"
        )
        model_validation_frame.grid(column=0, row=1)

        self.do_forecast_option = tk.IntVar(value=0)
        tk.Checkbutton(
            model_validation_frame,
            text="Do Forecast",
            offvalue=0,
            onvalue=1,
            variable=self.do_forecast_option,
            command=self.__open_other_entries,
        ).grid(column=0, row=0, columnspan=2)

        self.validation_option = tk.IntVar(value=0)
        self.random_percent_var = tk.IntVar(value=70)
        self.cross_val_var = tk.IntVar(value=5)
        tk.Radiobutton(
            model_validation_frame,
            text="No validation, use all data rows",
            value=0,
            variable=self.validation_option,
            command=self.__open_other_entries,
        ).grid(column=0, row=1, columnspan=2, sticky=tk.W)
        tk.Radiobutton(
            model_validation_frame,
            text="Random percent",
            value=1,
            variable=self.validation_option,
            command=self.__open_other_entries,
        ).grid(column=0, row=2, sticky=tk.W)
        self.cv_entry_1 = tk.Radiobutton(
            model_validation_frame,
            text="K-fold cross-validation",
            value=2,
            variable=self.validation_option,
            command=self.__open_other_entries,
        )
        self.cv_entry_1.grid(column=0, row=3, sticky=tk.W)
        self.cv_entry_2 = tk.Radiobutton(
            model_validation_frame,
            text="Leave one out cross-validation",
            value=3,
            variable=self.validation_option,
            command=self.__open_other_entries,
        )
        self.cv_entry_2.grid(column=0, row=4, columnspan=2, sticky=tk.W)
        self.random_percent_entry = ttk.Entry(
            model_validation_frame, textvariable=self.random_percent_var, width=8
        )
        self.random_percent_entry.grid(column=1, row=2)
        self.cv_value_entry = ttk.Entry(
            model_validation_frame, textvariable=self.cross_val_var, width=8
        )
        self.cv_value_entry.grid(column=1, row=3)

        # Customize Train Set
        customize_train_set_frame = ttk.LabelFrame(
            self.root, text="Customize Train Set"
        )
        customize_train_set_frame.grid(column=0, row=2)

        self.lookback_option = tk.IntVar(value=0)
        self.lookback_val_var = tk.IntVar(value="")  # type: ignore
        tk.Checkbutton(
            customize_train_set_frame,
            text="Lookback",
            offvalue=0,
            onvalue=1,
            variable=self.lookback_option,
            command=self.__open_other_entries,
        ).grid(column=0, row=0)
        self.lookback_entry = tk.Entry(
            customize_train_set_frame,
            textvariable=self.lookback_val_var,
            width=8,
            state=tk.DISABLED,
        )
        self.lookback_entry.grid(column=1, row=0)

        self.seasonal_lookback_option = tk.IntVar(value=0)
        self.seasonal_period_var = tk.IntVar(value="")  # type: ignore
        self.seasonal_val_var = tk.IntVar(value="")  # type: ignore
        tk.Checkbutton(
            customize_train_set_frame,
            text="Periodic Lookback",
            offvalue=0,
            onvalue=1,
            variable=self.seasonal_lookback_option,
            command=self.__open_other_entries,
        ).grid(column=0, row=1)
        self.seasonal_lookback_entry_1 = tk.Entry(
            customize_train_set_frame,
            textvariable=self.seasonal_period_var,
            width=9,
            state=tk.DISABLED,
        )
        self.seasonal_lookback_entry_1.grid(column=0, row=2)
        self.seasonal_lookback_entry_2 = tk.Entry(
            customize_train_set_frame,
            textvariable=self.seasonal_val_var,
            width=8,
            state=tk.DISABLED,
        )
        self.seasonal_lookback_entry_2.grid(column=1, row=2)

        self.scale_var = tk.StringVar(value="None")
        ttk.Label(customize_train_set_frame, text="Scale Type").grid(column=0, row=3)
        ttk.OptionMenu(
            customize_train_set_frame,
            self.scale_var,
            "None",
            "None",
            "StandardScaler",
            "MinMaxScaler",
        ).grid(column=1, row=3)

