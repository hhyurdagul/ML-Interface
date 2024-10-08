from PySide6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QFrame
)
from PySide6.QtCore import Qt
from typing import Callable

class PredictionComponent(QGroupBox):
    def __init__(self, title: str, get_result_data: Callable[[str], list[str]] = lambda _: list()):
        super().__init__(title)

        layout = QHBoxLayout()
        self.setLayout(layout)

        # Test Model Main
        test_model_main_frame = QGroupBox("Test Model")
        test_model_main_layout = QVBoxLayout()
        test_model_main_frame.setLayout(test_model_main_layout)

        prediction_count_layout = QHBoxLayout()
        prediction_count_layout.addWidget(QLabel("Prediction Count"))
        self.prediction_count = QLineEdit()
        self.prediction_count.setFixedWidth(100)
        prediction_count_layout.addWidget(self.prediction_count)
        result_values_button = QPushButton("Result Values")
        result_values_button.setFixedWidth(100)
        result_values_button.clicked.connect(self.show_result_values)
        prediction_count_layout.addWidget(result_values_button)
        test_model_main_layout.addLayout(prediction_count_layout)

        test_file_layout = QHBoxLayout()
        test_file_layout.addWidget(QLabel("Test File Path"))
        self.test_file_path = QLineEdit()
        self.test_file_path.setFixedWidth(100)
        self.test_file_path.setEnabled(False)
        test_file_layout.addWidget(self.test_file_path)
        get_test_set_button = QPushButton("Get Test Set")
        get_test_set_button.setFixedWidth(100)
        get_test_set_button.clicked.connect(self.read_test_data)
        test_file_layout.addWidget(get_test_set_button)
        test_model_main_layout.addLayout(test_file_layout)

        test_model_buttons_layout = QHBoxLayout()
        test_model_button = QPushButton("Test Model")
        test_model_button.clicked.connect(self.forecast)
        test_model_buttons_layout.addWidget(test_model_button)
        result_graph_button = QPushButton("Result Graph")
        result_graph_button.clicked.connect(self.show_result_graph)
        test_model_buttons_layout.addWidget(result_graph_button)
        test_model_main_layout.addLayout(test_model_buttons_layout)

        layout.addWidget(test_model_main_frame)

        # Test Model Metrics
        test_model_metrics_frame = QGroupBox("Test Metrics")
        test_model_metrics_layout = QVBoxLayout()
        test_model_metrics_frame.setLayout(test_model_metrics_layout)

        test_metric_names = ["R2:", "MAE:", "MAPE:", "SMAPE:"]
        self.test_metric_inputs = []
        for metric_name in test_metric_names:
            metric_layout = QHBoxLayout()
            metric_layout.addWidget(QLabel(metric_name))
            metric_input = QLineEdit()
            metric_input.setFixedWidth(100)
            metric_input.setEnabled(False)
            self.test_metric_inputs.append(metric_input)
            metric_layout.addWidget(metric_input)
            test_model_metrics_layout.addLayout(metric_layout)

        layout.addWidget(test_model_metrics_frame)

        self.get_result_data = get_result_data

    def show_result_values(self):
        # Implement this method
        pass

    def read_test_data(self):
        # Implement this method
        pass

    def forecast(self):
        # Implement this method
        pass

    def show_result_graph(self):
        # Implement this method
        pass
