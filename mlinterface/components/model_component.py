from PySide6.QtWidgets import QWidget, QGridLayout, QLabel, QLineEdit, QGroupBox, QComboBox, QCheckBox, QPushButton
from PySide6.QtCore import Signal
from mlinterface.components.variables import QIntLineEdit, QFloatLineEdit

class RandomForestWidget(QGroupBox):
    def __init__(self):
        super().__init__("Random Forest")
        layout = QGridLayout(self)
        self.parameters = {
            "n_estimators": QIntLineEdit("100", 1, 1000),
            "max_depth": QIntLineEdit("10", 1, 100),
            "min_samples_split": QIntLineEdit("2", 2, 10),
            "min_samples_leaf": QIntLineEdit("1", 1, 10),
        }
        for i, (name, widget) in enumerate(self.parameters.items()):
            layout.addWidget(QLabel(f"{name.replace('_', ' ').title()}:"), i, 0)
            widget.setFixedWidth(100)
            layout.addWidget(widget, i, 1)
        self.setLayout(layout)

    def toggle_params(self):
        for widget in self.parameters.values():
            widget.setEnabled(not widget.isEnabled())

class XGBoostWidget(QGroupBox):
    def __init__(self):
        super().__init__("XGBoost")
        layout = QGridLayout(self)
        self.parameters = {
            "n_estimators": QIntLineEdit("100", 1, 1000),
            "max_depth": QIntLineEdit("6", 1, 100),
            "learning_rate": QFloatLineEdit("0.3", 0.01, 1),
        }
        for i, (name, widget) in enumerate(self.parameters.items()):
            layout.addWidget(QLabel(f"{name.replace('_', ' ').title()}:"), i, 0)
            widget.setFixedWidth(100)
            layout.addWidget(widget, i, 1)
        self.setLayout(layout)

    def toggle_params(self):
        for widget in self.parameters.values():
            widget.setEnabled(not widget.isEnabled())

class ModelComponent(QGroupBox):
    def __init__(self, title: str):
        super().__init__(title)
        self.main_layout = QGridLayout()

        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(["RandomForest", "XGBoost"])
        self.algorithm_combo.currentIndexChanged.connect(lambda x: self.on_algorithm_changed(x))

        self.main_layout.addWidget(QLabel("Algorithm:"), 0, 0)
        self.main_layout.addWidget(self.algorithm_combo, 0, 1)

        self.optimization_checkbox = QCheckBox("Use Optimization")

        self.current_algorithm = RandomForestWidget()
        self.on_algorithm_changed(0)

        self.main_layout.addWidget(self.optimization_checkbox, 0, 2)
        self.create_model_button = QPushButton("Create Model")
        self.save_model_button = QPushButton("Save Model")
        self.load_model_button = QPushButton("Load Model")

        self.main_layout.addWidget(self.create_model_button, 1, 2)
        self.main_layout.addWidget(self.save_model_button, 2, 2)
        self.main_layout.addWidget(self.load_model_button, 3, 2)

        self.setLayout(self.main_layout)


    def on_algorithm_changed(self, index):
        if self.current_algorithm:
            self.current_algorithm.deleteLater()

        if index == 0:
           self.current_algorithm = RandomForestWidget()
        else:
           self.current_algorithm = XGBoostWidget()

        self.optimization_checkbox.stateChanged.connect(self.current_algorithm.toggle_params)
        self.main_layout.addWidget(self.current_algorithm, 1, 0, 3, 2)