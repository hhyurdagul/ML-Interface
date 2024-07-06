import tkinter as tk
from tkinter import ttk
from typing import Any
from mlinterface.gui.components.variables import change_state


class ModelComponent:
    def __init__(
        self, parent: ttk.Frame, text: str, model_parameters: dict[str, Any]
    ) -> None:
        self.root = ttk.LabelFrame(parent, text=text)

        model_frame = ttk.Labelframe(self.root, text="Model Frame")
        model_frame.grid(column=1, row=0)

        self.do_optimization = tk.IntVar(value=0)

        ttk.Radiobutton(
            model_frame,
            text="No Optimization",
            variable=self.do_optimization,
            value=0,
            command=lambda: change_state(1, parameter_entries),
        ).grid(column=0, row=0)
        ttk.Radiobutton(
            model_frame,
            text="Do Optimization",
            variable=self.do_optimization,
            value=1,
            command=lambda: change_state(0, parameter_entries),
        ).grid(column=2, row=0)

        model_parameters_frame = ttk.LabelFrame(model_frame, text="Parameters")
        model_parameters_frame.grid(column=0, row=1, columnspan=3)

        parameter_names = [
            "N Estimators",
            "Max Depth",
            "Min Samp. Split",
            "Min Samp. Leaf",
        ]
        self.parameters = [
            tk.IntVar(value=100),
            tk.IntVar(value=100),
            tk.IntVar(value=2),
            tk.IntVar(value=1),
        ]

        parameter_entries = [
            ttk.Entry(
                model_parameters_frame,
                textvariable=var,
                width=8,
            )
            for var in self.parameters
        ]

        for i, j in enumerate(parameter_names):
            ttk.Label(model_parameters_frame, text=f"{j}:", width=12).grid(
                column=0, row=i, sticky="w"
            )
            parameter_entries[i].grid(column=1, row=i, padx=2, pady=2, sticky=tk.W)

        ttk.Button(model_frame, text="Create Model", command=self.create_model).grid(
            column=0, row=4
        )
        ttk.Button(model_frame, text="Save Model", command=self.save_model).grid(
            column=1, row=4
        )
        ttk.Button(model_frame, text="Load Model", command=self.load_model).grid(
            column=2, row=4
        )

    def grid(self, column: int, row: int) -> "ModelComponent":
        self.root.grid(column=column, row=row)
        return self
