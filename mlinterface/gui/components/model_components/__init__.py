import tkinter as tk
from tkinter import ttk
from typing import Callable
from .random_forest import RandomForestComponent


class ModelComponent:
    def __init__(
        self,
        parent: ttk.Frame,
        text: str,
        model_type: str,
        create_model_func: Callable[[], None],
        save_model_func: Callable[[], None],
        load_model_func: Callable[[], None],
    ) -> None:
        self.root = ttk.LabelFrame(parent, text=text)

        if model_type == "RandomForest":
            self.model_parameters_frame = RandomForestComponent(self.root)
        else:
            self.model_parameters_frame = RandomForestComponent(self.root)

        self.do_optimization = tk.IntVar(value=0)
        ttk.Radiobutton(
            self.root,
            text="No Optimization",
            variable=self.do_optimization,
            value=0,
            command=self.model_parameters_frame.enable_parameter_entries,
        ).grid(column=0, row=0)
        ttk.Radiobutton(
            self.root,
            text="Do Optimization",
            variable=self.do_optimization,
            value=1,
            command=self.model_parameters_frame.disable_parameter_entries,
        ).grid(column=2, row=0)

        ttk.Button(self.root, text="Create Model", command=create_model_func).grid(
            column=0, row=4
        )
        ttk.Button(self.root, text="Save Model", command=save_model_func).grid(
            column=1, row=4
        )
        ttk.Button(self.root, text="Load Model", command=load_model_func).grid(
            column=2, row=4
        )

    def get_params(self) -> dict[str, int]:
        return self.model_parameters_frame.get_parameters()

    def set_params(self, params: dict[str, int]) -> None:
        self.model_parameters_frame.set_parameters(params)

    def grid(self, column: int, row: int) -> "ModelComponent":
        self.root.grid(column=column, row=row)
        return self
