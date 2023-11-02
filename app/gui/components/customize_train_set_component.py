import tkinter as tk
from tkinter import ttk

from typing import Any, Union
from ..backend import ScalerHandler, LookbackHandler
from .utils import popupmsg


class CustomizeTrainSetComponent:
    def __init__(
        self,
        root: ttk.Frame,
        scaler_handler: ScalerHandler,
        lookback_handler: LookbackHandler,
    ) -> None:
        self.scaler_handler = scaler_handler
        self.lookback_handler = lookback_handler

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

    def calculate_sliding(self):
        self.sliding = (
            self.lookback_option.get() + 2 * self.seasonal_lookback_option.get() - 1
        )

    def check_errors(self) -> bool:
        if self.lookback_option.get() and self.lookback_val_var.get() <= 0:
            return popupmsg("Enter a valid lookback value")

        if self.seasonal_lookback_option.get() and (
            self.seasonal_val_var.get() <= 0 or self.seasonal_period_var.get() <= 0
        ):
            return popupmsg("Enter valid periodic lookback values")

        return True

    def get_params(self) -> dict[str, Union[str, int]]:
        return {
            "lookback_option": self.lookback_option.get(),
            "lookback_value": self.lookback_val_var.get(),
            "seasonal_lookback_option": self.seasonal_lookback_option.get(),
            "seasonal_period": self.seasonal_period_var.get(),
            "seasonal_value": self.seasonal_val_var.get(),
            "sliding": self.lookback_option.get()
            + 2 * self.seasonal_lookback_option.get()
            - 1,
            "scale_type": self.scale_var.get(),
        }

    def set_params(self, params: dict[str, Any]) -> None:
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

    def save_files(self, path: str) -> None:
        self.lookback_handler.save_lasts(
            path,
            bool(self.lookback_option.get()),
            bool(self.seasonal_lookback_option.get()),
        )

        if self.scale_var.get() != "None":
            self.scaler_handler.save_scalers(path)

    def load_files(self, path: str) -> None:
        if self.lookback_option.get() == 1 or self.seasonal_lookback_option.get():
            self.lookback_handler.load_lasts(path)

        if self.scale_var.get() != "None":
            self.scaler_handler.load_scalers(path)

    def __open_entries(self) -> None:
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
