import tkinter as tk
from tkinter import ttk
from typing import Any

from mlinterface.gui.components.variables import GenericIntVar, change_state


class PreprocessingComponent:
    def __init__(self, parent: ttk.Frame, text: str) -> None:
        self.root = ttk.LabelFrame(parent, text=text)

        self.train_data_size = tk.IntVar(value=100)
        self.scale_type = tk.StringVar(value="None")

        self.lookback_option = tk.IntVar(value=0)
        self.lookback_value = GenericIntVar(value="")

        self.seasonal_lookback_option = tk.IntVar(value=0)
        self.seasonal_lookback_value = GenericIntVar(value="")
        self.seasonal_lookback_frequency = GenericIntVar(value="")

        ttk.Label(self.root, text="Train data size:", width=12).grid(
            column=0, row=0, sticky="w"
        )
        ttk.Entry(self.root, textvariable=self.train_data_size, width=8).grid(
            column=1, row=0, sticky="w"
        )

        ttk.Label(self.root, text="Scale type:", width=12).grid(
            column=0, row=1, sticky="w"
        )
        ttk.OptionMenu(
            self.root,
            self.scale_type,
            "None",
            "None",
            "MinMax",
            "Standard",
        ).grid(column=1, row=1, sticky="w")

        ttk.Checkbutton(
            self.root,
            text="Lookback",
            offvalue=0,
            onvalue=1,
            variable=self.lookback_option,
            command=lambda: change_state(
                self.lookback_option.get(), [lookback_entry], [self.lookback_value]
            ),
            width=12,
        ).grid(column=0, row=2, columnspan=2)

        ttk.Label(self.root, text="Value:", width=12).grid(column=0, row=3, sticky="w")
        lookback_entry = ttk.Entry(
            self.root,
            textvariable=self.lookback_value,
            width=8,
            state=tk.DISABLED,
        )
        lookback_entry.grid(column=1, row=3, sticky="w")

        ttk.Checkbutton(
            self.root,
            text="Periodic Lookback",
            offvalue=0,
            onvalue=1,
            variable=self.seasonal_lookback_option,
            command=lambda: change_state(
                self.seasonal_lookback_option.get(),
                [seasonal_lookback_value_entry, seasonal_lookback_frequency_entry],
                [self.seasonal_lookback_value, self.seasonal_lookback_frequency],
            ),
            width=13,
        ).grid(column=0, row=4, columnspan=2)

        ttk.Label(self.root, text="Value:", width=12).grid(column=0, row=5, sticky="w")
        seasonal_lookback_value_entry = ttk.Entry(
            self.root,
            textvariable=self.seasonal_lookback_value,
            state=tk.DISABLED,
            width=8,
        )
        seasonal_lookback_value_entry.grid(column=1, row=5, sticky="w")

        ttk.Label(self.root, text="Frequency:", width=12).grid(
            column=0, row=6, sticky="w"
        )
        seasonal_lookback_frequency_entry = ttk.Entry(
            self.root,
            textvariable=self.seasonal_lookback_frequency,
            state=tk.DISABLED,
            width=8,
        )
        seasonal_lookback_frequency_entry.grid(column=1, row=6, sticky="w")

    def get_params(self) -> dict[str, Any]:
        data = {
            "train_data_size": self.train_data_size.get(),
            "scale_type": self.scale_type.get(),
            "lookback_option": self.lookback_option.get(),
            "lookback_value": self.lookback_value.get(),
            "seasonal_lookback_option": self.seasonal_lookback_option.get(),
            "seasonal_lookback_value": self.seasonal_lookback_value.get(),
            "seasonal_lookback_frequency": self.seasonal_lookback_frequency.get(),
        }
        return data

    def set_params(self, data: dict[str, Any]) -> None:
        self.train_data_size.set(data["train_data_size"])
        self.scale_type.set(data["scale_type"])
        self.lookback_option.set(data["lookback_option"])
        self.lookback_value.set(
            "" if data["lookback_option"] == 0 else data["lookback_value"]
        )
        self.seasonal_lookback_option.set(data["seasonal_lookback_option"])
        self.seasonal_lookback_value.set(
            ""
            if data["seasonal_lookback_option"] == 0
            else data["seasonal_lookback_value"]
        )
        self.seasonal_lookback_frequency.set(
            ""
            if data["seasonal_lookback_option"] == 0
            else data["seasonal_lookback_frequency"]
        )

    def check_errors(self) -> None:
        if self.train_data_size.get() < 0:
            raise ValueError("Train data size must be greater than 0")

        if self.train_data_size.get() > 100:
            raise ValueError("Train data size must be less than 100")

        if self.lookback_option.get() == 1 and self.lookback_value.get() == "":
            raise ValueError("Lookback value must be specified")

        if self.seasonal_lookback_option.get() == 1:
            if self.seasonal_lookback_value.get() == "":
                raise ValueError("Seasonal lookback value must be specified")
            if self.seasonal_lookback_frequency.get() == "":
                raise ValueError("Seasonal lookback frequency must be specified")

    def grid(self, column: int, row: int) -> "PreprocessingComponent":
        self.root.grid(column=column, row=row)
        return self
