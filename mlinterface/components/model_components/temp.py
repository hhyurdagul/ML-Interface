from PySide6.QtWidgets import QWidget, QGridLayout, QLabel, QLineEdit, QGroupBox, QComboBox, QCheckBox
from PySide6.QtCore import Signal

class RandomForestWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QGridLayout(self)
        self.parameters = {
            "n_estimators": QLineEdit("100"),
            "max_depth": QLineEdit("100"),
            "min_samples_split": QLineEdit("2"),
            "min_samples_leaf": QLineEdit("1"),
        }
        for i, (name, widget) in enumerate(self.parameters.items()):
            layout.addWidget(QLabel(f"{name.replace('_', ' ').title()}:"), i, 0)
            layout.addWidget(widget, i, 1)

    def toggle_params(self):
        for widget in self.parameters.values():
            widget.setEnabled(not widget.isEnabled())

class XGBoostWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QGridLayout(self)
        self.parameters = {
            "n_estimators": QLineEdit("100"),
            "max_depth": QLineEdit("6"),
            "learning_rate": QLineEdit("0.3"),
            "subsample": QLineEdit("1"),
        }
        for i, (name, widget) in enumerate(self.parameters.items()):
            layout.addWidget(QLabel(f"{name.replace('_', ' ').title()}:"), i, 0)
            layout.addWidget(widget, i, 1)

    def toggle_params(self):
        for widget in self.parameters.values():
            widget.setEnabled(not widget.isEnabled())

class ModelComponent(QWidget):
    def __init__(self):
        super().__init__()
        self.root = QGroupBox("Model Parameters")
        self.main_layout = QGridLayout()
        self.root.setLayout(self.main_layout)

        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(["RandomForest", "XGBoost"])
        self.algorithm_combo.currentIndexChanged.connect(lambda x: self.on_algorithm_changed(x))

        self.main_layout.addWidget(QLabel("Algorithm:"), 0, 0)
        self.main_layout.addWidget(self.algorithm_combo, 0, 1)

        self.optimization_checkbox = QCheckBox("Use Optimization")

        self.current_algorithm = RandomForestWidget()
        self.on_algorithm_changed(0)

        self.optimization_checkbox.stateChanged.connect(lambda x: print(x))

        self.main_layout.addWidget(self.optimization_checkbox, 1, 0)


    def on_algorithm_changed(self, index):
        if self.current_algorithm:
            self.current_algorithm.deleteLater()

        if index == 0:
           self.current_algorithm = RandomForestWidget()
        else:
           self.current_algorithm = XGBoostWidget()

        self.optimization_checkbox.stateChanged.connect(self.current_algorithm.toggle_params)
        self.main_layout.addWidget(self.current_algorithm, 2, 0, 1, 2)
