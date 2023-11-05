import tkinter as tk
from tkinter import ttk

from .utils import popupmsg
from ..backend import ModelHandler

from typing import List, Any, Callable


class ModelComponent:
    def __init__(
        self,
        root: ttk.Frame,
        model_handler: ModelHandler,
        create_model: Callable,
        save_model: Callable,
        load_model: Callable,
    ) -> None:
        self.root = root
        self.model_handler = model_handler

        self.grid_option_var = tk.IntVar(value=0)
        self.interval_var = tk.IntVar(value=3)
        self.gs_cross_val_var = tk.IntVar(value=3)

        model_frame = ttk.Labelframe(self.root, text="Model Frame")
        model_frame.grid(column=1, row=0)

        parameter_names = ["N Estimators", "Max Depth", "Learning Rate"]
        self.parameters = [
            tk.IntVar(value=100),
            tk.IntVar(value=6),
            tk.DoubleVar(value=0.3),
        ]
        self.optimization_parameters: List[List[Any]] = [
            [tk.IntVar(value=75), tk.IntVar(value=150)],
            [tk.IntVar(value=5), tk.IntVar(value=10)],
            [tk.DoubleVar(value=0.2), tk.DoubleVar(value=1)],
        ]

        ttk.Label(model_frame, text="Current").grid(column=1, row=0)
        ttk.Label(model_frame, text="----- Search Range -----").grid(
            column=2, row=0, columnspan=2
        )

        for i, j in enumerate(parameter_names):
            ttk.Label(model_frame, text=j + ":").grid(column=0, row=i + 1),

        self.model_parameters_frame_options: List[List[ttk.Entry]] = [
            [
                ttk.Entry(
                    model_frame,
                    textvariable=self.parameters[i],
                    state=tk.DISABLED,
                    width=9,
                ),
                ttk.Entry(
                    model_frame,
                    textvariable=self.optimization_parameters[i][0],
                    state=tk.DISABLED,
                    width=9,
                ),
                ttk.Entry(
                    model_frame,
                    textvariable=self.optimization_parameters[i][1],
                    state=tk.DISABLED,
                    width=9,
                ),
            ]
            for i, j in enumerate(parameter_names)
        ]

        for i, entry_list in enumerate(self.model_parameters_frame_options):
            entry_list[0].grid(column=1, row=i + 1, padx=2, pady=2, sticky=tk.W)
            entry_list[1].grid(column=2, row=i + 1, padx=2, pady=2)
            entry_list[2].grid(column=3, row=i + 1, padx=2, pady=2)

        last_row = len(self.model_parameters_frame_options) + 1

        tk.Checkbutton(
            model_frame,
            text="Do grid search for optimal parameters",
            offvalue=0,
            onvalue=1,
            variable=self.grid_option_var,
            command=self.__open_entries,
        ).grid(column=0, row=last_row, columnspan=3)

        ttk.Label(model_frame, text="Interval:").grid(column=0, row=last_row + 1)
        self.interval_entry = ttk.Entry(
            model_frame,
            textvariable=self.interval_var,
            width=8,
            state=tk.DISABLED,
        )
        self.interval_entry.grid(column=1, row=last_row + 1, pady=2)

        ttk.Label(
            model_frame,
            text="Folds:",
        ).grid(column=2, row=last_row + 1)

        self.gs_cross_val_entry = tk.Entry(
            model_frame,
            textvariable=self.gs_cross_val_var,
            state=tk.DISABLED,
            width=8,
        )
        self.gs_cross_val_entry.grid(column=3, row=last_row + 1)

        ttk.Button(model_frame, text="Create Model", command=create_model).grid(
            column=0, row=last_row + 2
        )
        ttk.Button(model_frame, text="Save Model", command=save_model).grid(
            column=1, row=last_row + 2, columnspan=2
        )
        ttk.Button(model_frame, text="Load Model", command=load_model).grid(
            column=3, row=last_row + 2
        )

    def check_errors(self) -> bool:
        if self.grid_option_var.get() and self.interval_var.get() < 1:
            return popupmsg("Enter a valid Interval for grid search")

        if self.grid_option_var.get() and self.gs_cross_val_var.get() < 2:
            return popupmsg(
                "Enter a valid Cross Validation fold for grid search (Above 2)"
            )

        return True

    def get_model_params(self) -> dict[str, Any]:
        return {
            "n_estimators": self.parameters[0].get(),
            "max_depth": self.parameters[1].get(),
            "learning_rate": self.parameters[2].get(),
        }

    def get_params(self) -> dict[str, Any]:
        return self.model_handler.model_params

    def set_params(self, params: dict[str, Any]) -> None:
        self.parameters[0].set(params["n_estimators"])
        self.parameters[1].set(params["max_depth"])
        self.parameters[2].set(params["learning_rate"])

    def save_files(self, path: str) -> None:
        self.model_handler.save_files(path)
        

    def load_files(self, path: str) -> bool:
        self.model_handler.load_model(path)
        if not self.model_handler.model_created:
            return popupmsg("Model file not found")
        return True

    def get_grid_params(self):
        return {
            "n_estimators": [
                self.optimization_parameters[0][0].get(),
                self.optimization_parameters[0][1].get(),
            ],
            "max_depth": [
                self.optimization_parameters[1][0].get(),
                self.optimization_parameters[1][1].get(),
            ],
            "learning_rate": [
                self.optimization_parameters[2][0].get(),
                self.optimization_parameters[2][1].get(),
            ],
            "interval": self.interval_var.get(),
            "cv": self.gs_cross_val_var.get(),
        }

    def __open_entries(self) -> None:
        opt = self.grid_option_var.get()
        self.interval_entry["state"] = tk.DISABLED
        self.gs_cross_val_entry["state"] = tk.DISABLED

        if opt:
            self.interval_entry["state"] = tk.NORMAL
            self.gs_cross_val_entry["state"] = tk.NORMAL

        to_open = []
        for i in self.model_parameters_frame_options:
            i[1]["state"] = tk.DISABLED
            i[2]["state"] = tk.DISABLED
            i[3]["state"] = tk.DISABLED

        to_open = list(range(len(self.parameters)))
        if opt == 1:
            self.interval_entry["state"] = tk.NORMAL
            for index in to_open:
                self.model_parameters_frame_options[index][2]["state"] = tk.NORMAL
                self.model_parameters_frame_options[index][3]["state"] = tk.NORMAL
        else:
            for index in to_open:
                self.model_parameters_frame_options[index][1]["state"] = tk.NORMAL
