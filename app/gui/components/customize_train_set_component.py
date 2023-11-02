import os
import numpy as np
import tkinter as tk
from tkinter import ttk
import pickle
from .utils import popupmsg


class CustomizeTrainSetComponent:
    def __init__(self, root):
        self.root = root
        self.lookback_option = tk.IntVar(value=0)
        self.lookback_val_var = tk.IntVar(value=0)
        self.seasonal_lookback_option = tk.IntVar(value=0)
        self.seasonal_period_var = tk.IntVar(value=0)
        self.seasonal_val_var = tk.IntVar(value=0)
        self.sliding = -1
        self.scale_var = tk.StringVar(value="None")

        customize_train_set_frame = ttk.LabelFrame(
            self.root, text="Customize Train Set"
        )
        customize_train_set_frame.grid(column=0, row=2)

        tk.Checkbutton(
            customize_train_set_frame,
            text="Lookback",
            offvalue=0,
            onvalue=1,
            variable=self.lookback_option,
            command=self.__open_entries,
        ).grid(column=0, row=0)
        self.lookback_entry = tk.Entry(
            customize_train_set_frame,
            textvariable=self.lookback_val_var,
            width=8,
            state=tk.DISABLED,
        )
        self.lookback_entry.grid(column=1, row=0)

        tk.Checkbutton(
            customize_train_set_frame,
            text="Periodic Lookback",
            offvalue=0,
            onvalue=1,
            variable=self.seasonal_lookback_option,
            command=self.__open_entries,
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

        ttk.Label(customize_train_set_frame, text="Scale Type").grid(column=0, row=3)
        ttk.OptionMenu(
            customize_train_set_frame,
            self.scale_var,
            "None",
            "None",
            "StandardScaler",
            "MinMaxScaler",
        ).grid(column=1, row=3)

    def check_errors(self) -> bool:
        if self.lookback_option.get() and self.lookback_val_var.get() <= 0:
            return popupmsg("Enter a valid lookback value")

        if self.seasonal_lookback_option.get() and (
            self.seasonal_val_var.get() <= 0 or self.seasonal_period_var.get() <= 0
        ):
            return popupmsg("Enter valid periodic lookback values")

        return True

    def get_params(self):
        return {
            "lookback_option": self.lookback_option.get(),
            "lookback_value": self.lookback_val_var.get(),
            "seasonal_lookback_option": self.seasonal_lookback_option.get(),
            "seasonal_period": self.seasonal_period_var.get(),
            "seasonal_value": self.seasonal_val_var.get(),
            "sliding": self.lookback_option + 2 * self.seasonal_lookback_option - 1,
            "scale_type": self.scale_var.get(),
        }

    def set_params(self, params):
        self.lookback_option.set(params.get("lookback_option", 0))
        self.sliding = params.get("sliding", -1)
        if self.lookback_option.get() == 1:
            self.lookback_val_var.set(params.get("lookback_value", 7))

        self.seasonal_lookback_option.set(params.get("seasonal_lookback_option", 0))

        if self.seasonal_lookback_option.get() == 1:
            self.seasonal_period_var.set(params.get("seasonal_period", 8))
            self.seasonal_val_var.set(params.get("seasonal_value", 7))

        self.scale_var.set(params.get("scale_type", "None"))

        self.__open_entries()

    def save_files(self, path: str, files: dict[str, object]):
        last_values_path = f"{path}/last_values.npy"
        slv_path = f"{path}/seasonal_last_values.py"

        if self.lookback_option.get() == 1:
            with open(last_values_path, "wb") as outfile:
                np.save(outfile, files["last"])
        if self.seasonal_lookback_option.get() == 1:
            with open(slv_path, "wb") as outfile:
                np.save(outfile, files["seasonal_last"])

        fs_path = f"{path}/feature_scaler.pkl"
        ls_path = f"{path}/label_scaler.pkl"

        if self.scale_var.get() != "None":
            with open(fs_path, "wb") as fs, open(ls_path, "wb") as ls:
                pickle.dump(files["feature_scaler"], fs)
                pickle.dump(files["label_scaler"], ls)

    def load_files(self, path: str) -> dict[str:object]:
        files: dict[str:object] = {}

        last_values_path = f"{path}/last_values.npy"
        slv_path = f"{path}/seasonal_last_values.py"
        if self.lookback_option.get() == 1 and os.path.exists(last_values_path):
            with open(last_values_path, "rb") as last_values:
                files["last"] = np.load(last_values)
        if self.seasonal_lookback_option.get() == 1 and os.path.exists(slv_path):
            with open(slv_path, "rb") as slv:
                files["seasonal_last"] = np.load(slv)

        fs_path = f"{path}/feature_scaler.pkl"
        ls_path = f"{path}/label_scaler.pkl"

        if (
            self.scale_var.get() != "None"
            and os.path.exists(fs_path)
            and os.path.exists(ls_path)
        ):
            with open(fs_path, "rb") as fs, open(ls_path, "rb") as ls:
                files["feature_scaler"] = pickle.load(fs)
                files["label_scaler"] = pickle.load(ls)
        return files

    def __open_entries(self):
        if self.lookback_option.get():
            self.lookback_entry["state"] = tk.NORMAL
        else:
            self.lookback_entry["state"] = tk.DISABLED
        if self.seasonal_lookback_option.get():
            self.seasonal_lookback_entry_1["state"] = tk.NORMAL
            self.seasonal_lookback_entry_2["state"] = tk.NORMAL
        else:
            self.seasonal_lookback_entry_1["state"] = tk.DISABLED
            self.seasonal_lookback_entry_2["state"] = tk.DISABLED
