import tkinter as tk
from tkinter import ttk
from typing import Callable

from mlinterface.gui.components.variables import GenericIntVar, GenericFloatVar


class PredictionComponent:
    def __init__(self, parent: ttk.Frame, text: str, get_result_data: Callable) -> None:
        self.root = ttk.LabelFrame(parent, text=text)

        ## Test Model Main
        test_model_main_frame = ttk.LabelFrame(self.root, text="Test Model")
        test_model_main_frame.grid(column=0, row=0)

        self.prediction_count = GenericIntVar(value="")  # type: ignore
        ttk.Label(test_model_main_frame, text="Prediction Count", width=12).grid(
            column=0, row=0, sticky="w"
        )
        ttk.Entry(
            test_model_main_frame, textvariable=self.prediction_count, width=8
        ).grid(column=1, row=0, sticky="w")
        ttk.Button(
            test_model_main_frame,
            text="Result Values",
            command=self.show_result_values,
            width=8,
        ).grid(column=2, row=0, sticky="w")

        test_file_path = tk.StringVar(value="")
        ttk.Label(test_model_main_frame, text="Test File Path").grid(column=0, row=1)
        ttk.Entry(test_model_main_frame, textvariable=test_file_path).grid(
            column=1, row=1
        )
        ttk.Button(
            test_model_main_frame,
            text="Get Test Set",
            command=lambda: self.read_test_data(test_file_path),
        ).grid(column=2, row=1)

        ttk.Button(
            test_model_main_frame, text="Test Model", command=self.forecast
        ).grid(column=2, row=3)
        ttk.Button(
            test_model_main_frame,
            text="Result Graph",
            command=self.show_result_graph,
        ).grid(column=0, row=4, columnspan=3)

        ## Test Model Metrics
        test_model_metrics_frame = ttk.LabelFrame(self.root, text="Test Metrics")
        test_model_metrics_frame.grid(column=1, row=0)

        test_metric_names: list[str] = ["R2", "MAE", "MAPE", "SMAPE"]
        self.test_metric_vars = [
            GenericFloatVar(value="") for _ in range(len(test_metric_names))
        ]
        for i, j in enumerate(test_metric_names):
            ttk.Label(test_model_metrics_frame, text=j, width=12).grid(column=0, row=i)
            ttk.Entry(
                test_model_metrics_frame, textvariable=self.test_metric_vars[i], width=8
            ).grid(column=1, row=i)

    def grid(self, column: int, row: int) -> "PredictionComponent":
        self.root.grid(column=column, row=row)
        return self
