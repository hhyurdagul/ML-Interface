import tkinter as tk
from tkinter import ttk
from .utils import popupmsg

class ModelValidationComponent:
    def __init__(self, root):
        self.do_forecast_option = tk.IntVar(value=0)
        self.validation_option = tk.IntVar(value=0)
        self.random_percent_var = tk.IntVar(value=70)
        self.cross_val_var = tk.IntVar(value=5)

        frame_root = ttk.Labelframe(root, text="Model testing and validation")
        frame_root.grid(column=0, row=1)

        tk.Checkbutton(
            frame_root,
            text="Do Forecast",
            offvalue=0,
            onvalue=1,
            variable=self.do_forecast_option,
            command=self.__open_entries,
        ).grid(column=0, row=0, columnspan=2)

        tk.Radiobutton(
            frame_root,
            text="No validation, use all data rows",
            value=0,
            variable=self.validation_option,
            command=self.__open_entries,
        ).grid(column=0, row=1, columnspan=2, sticky=tk.W)

        tk.Radiobutton(
            frame_root,
            text="Random percent",
            value=1,
            variable=self.validation_option,
            command=self.__open_entries,
        ).grid(column=0, row=2, sticky=tk.W)

        self.cv_entry_1 = tk.Radiobutton(
            frame_root,
            text="K-fold cross-validation",
            value=2,
            variable=self.validation_option,
            command=self.__open_entries,
        )

        self.cv_entry_1.grid(column=0, row=3, sticky=tk.W)

        self.cv_entry_2 = tk.Radiobutton(
            frame_root,
            text="Leave one out cross-validation",
            value=3,
            variable=self.validation_option,
            command=self.__open_entries,
        )
        self.cv_entry_2.grid(column=0, row=4, columnspan=2, sticky=tk.W)

        self.random_percent_entry = ttk.Entry(
            frame_root, textvariable=self.random_percent_var, width=8
        )

        self.random_percent_entry.grid(column=1, row=2)

        self.cv_value_entry = ttk.Entry(
            frame_root, textvariable=self.cross_val_var, width=8
        )

        self.cv_value_entry.grid(column=1, row=3)

    def check_errors(self) -> bool:
        if self.random_percent_var.get() <= 0:
            return popupmsg("Enter a valid percent value")

        elif (self.validation_option.get() == 2 and self.cross_val_var.get() <= 1):
            return popupmsg("Enter a valid K-fold value (Above 2)")

        return True


    def get_save_dict(self):
        return {
            "do_forecast": self.do_forecast_option.get(),
            "validation_option": self.validation_option.get(),
            "random_percent": self.random_percent_var.get(),
            "k_fold_cv": self.cross_val_var.get(),
        }

    def set_params(self, params):
        self.do_forecast_option.set(params.get("do_forecast", 1))
        self.validation_option.set(params.get("validation_option", 0))
        if self.validation_option.get() == 1:
            self.random_percent_var.set(params.get("random_percent", 80))
        elif self.validation_option.get() == 2:
            self.cross_val_var.set(params.get("k_fold_cv", 5))

    def __open_entries(self):
        if not self.do_forecast_option.get():
            self.cv_entry_1["state"] = tk.NORMAL
            self.cv_entry_2["state"] = tk.NORMAL
        else:
            self.cv_entry_1["state"] = tk.DISABLED
            self.cv_entry_2["state"] = tk.DISABLED
        if self.validation_option.get() == 1:
            self.random_percent_entry["state"] = tk.NORMAL
        else:
            self.random_percent_entry["state"] = tk.DISABLED
        if self.validation_option.get() == 2:
            self.cv_value_entry["state"] = tk.NORMAL
        else:
            self.cv_value_entry["state"] = tk.DISABLED

