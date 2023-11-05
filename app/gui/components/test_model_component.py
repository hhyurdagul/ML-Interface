import tkinter as tk
from tkinter import ttk, filedialog
from pandastable import Table  # type: ignore
import matplotlib.pyplot as plt # type: ignore
from typing import Callable
from .utils import popupmsg
from ..backend import ForecastHandler


class TestModelComponent:
    def __init__(
        self,
        root: ttk.Frame,
        forecast_handler: ForecastHandler,
        forecast_func: Callable
    ) -> None:
        self.root = root
        self.data_handler = forecast_handler.model_handler.data_handler
        self.forecast_handler = forecast_handler
        self.forecast_num = tk.IntVar(value=0)

        test_model_frame = ttk.LabelFrame(self.root, text="Test Frame")
        test_model_frame.grid(column=1, row=1)

        ## Test Model Main
        test_model_main_frame = ttk.LabelFrame(test_model_frame, text="Test Model")
        test_model_main_frame.grid(column=0, row=0)

        ttk.Label(test_model_main_frame, text="# of Forecast").grid(column=0, row=0)
        ttk.Entry(test_model_main_frame, textvariable=self.forecast_num).grid(
            column=1, row=0
        )
        ttk.Button(
            test_model_main_frame, text="Values", command=self.show_result_values
        ).grid(column=2, row=0)

        test_file_path = tk.StringVar()
        ttk.Label(test_model_main_frame, text="Test File Path").grid(column=0, row=1)
        ttk.Entry(test_model_main_frame, textvariable=test_file_path).grid(
            column=1, row=1
        )
        ttk.Button(
            test_model_main_frame,
            text="Get Test Set",
            command=lambda: self.get_test_data(test_file_path),
        ).grid(column=2, row=1)

        ttk.Button(
            test_model_main_frame,
            text="Test Model",
            command=forecast_func,
        ).grid(column=2, row=3)
        ttk.Button(
            test_model_main_frame,
            text="Actual vs Forecast Graph",
            command=self.show_result_graph,
        ).grid(column=0, row=4, columnspan=3)

        ## Test Model Metrics
        test_model_metrics_frame = ttk.LabelFrame(test_model_frame, text="Test Metrics")
        test_model_metrics_frame.grid(column=1, row=0)

        test_metrics = ["NMSE", "RMSE", "MAE", "MAPE", "SMAPE"]
        self.test_metrics_vars = [tk.Variable() for _ in range(len(test_metrics))]
        for i, j in enumerate(test_metrics):
            ttk.Label(test_model_metrics_frame, text=j).grid(column=0, row=i)
            ttk.Entry(
                test_model_metrics_frame, textvariable=self.test_metrics_vars[i]
            ).grid(column=1, row=i)

    def get_test_data(self, file_path):
        path = filedialog.askopenfilename(
            filetypes=[
                ("Csv Files", "*.csv"),
                ("Xlsx Files", "*.xlsx"),
                ("Xlrd Files", ".xls"),
            ]
        )
        if not path:
            return
        file_path.set(path)
        self.data_handler.read_test_data(path)

    def forecast(self):
        forecast_num = self.forecast_num.get()
        if not self.data_handler.test_df_read:
            popupmsg("Read a test data")
            return
        if not forecast_num:
            popupmsg("Enter a valid forecast number")
            return

        self.forecast_handler.forecast(forecast_num)
        for i in range(len(self.test_metrics_vars)):
            self.test_metrics_vars[i].set(self.forecast_handler.loss[i])


    def show_result_values(self):
        df = self.forecast_handler.result
        top = tk.Toplevel(self.root)
        pt = Table(top, dataframe=df, editable=False)
        pt.show()

    def show_result_graph(self):
        plt.plot(self.forecast_handler.result)
        plt.legend(["test", "pred"], loc="upper left")
        plt.show()
