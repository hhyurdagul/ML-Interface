import tkinter as tk
from tkinter import ttk


class RandomForestComponent:
    def __init__(self, parent: ttk.LabelFrame) -> None:
        root = ttk.LabelFrame(parent, text="Parameters")
        root.grid(column=0, row=1, columnspan=3)

        self.parameters: dict[str, tk.IntVar] = {
            "n_estimators": tk.IntVar(value=100),
            "max_depth": tk.IntVar(value=100),
            "min_samples_split": tk.IntVar(value=2),
            "min_samples_leaf": tk.IntVar(value=1),
        }

        parameter_names = [
            "N Estimators",
            "Max Depth",
            "Min Samples Split",
            "Min Samples Leaf",
        ]

        self.parameter_entries = [
            ttk.Entry(
                root,
                textvariable=var,
                width=8,
            )
            for var in self.parameters.values()
        ]

        for i, j in enumerate(parameter_names):
            ttk.Label(root, text=f"{j}:", width=12).grid(column=0, row=i, sticky="w")
            self.parameter_entries[i].grid(column=1, row=i, padx=2, pady=2, sticky=tk.W)

    def disable_parameter_entries(self) -> None:
        for entry in self.parameter_entries:
            entry["state"] = tk.DISABLED

    def enable_parameter_entries(self) -> None:
        for entry in self.parameter_entries:
            entry["state"] = tk.NORMAL

    def get_parameters(self) -> dict[str, int]:
        return {i: j.get() for i, j in self.parameters.items()}

    def set_parameters(self, params: dict[str, int]) -> None:
        for i, j in self.parameters.items():
            j.set(params[i])
