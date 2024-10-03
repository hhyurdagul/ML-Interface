from PySide6.QtWidgets import (QWidget, QLabel, QLineEdit, QComboBox, QCheckBox,
                               QGridLayout, QGroupBox)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIntValidator
from typing import Any, Dict

class PreprocessingComponent(QWidget):
    paramsChanged = Signal()

    def __init__(self, text: str) -> None:
        super().__init__()
        self.root = QGroupBox(text)
        layout = QGridLayout()
        self.root.setLayout(layout)

        self.train_data_size = QLineEdit("100")
        self.train_data_size.setValidator(QIntValidator(0, 100))
        self.scale_type = QComboBox()
        self.scale_type.addItems(["None", "MinMax", "Standard"])

        self.lookback_option = QCheckBox("Lookback")
        self.lookback_value = QLineEdit()
        self.lookback_value.setEnabled(False)

        self.seasonal_lookback_option = QCheckBox("Periodic Lookback")
        self.seasonal_lookback_value = QLineEdit()
        self.seasonal_lookback_value.setEnabled(False)
        self.seasonal_lookback_frequency = QLineEdit()
        self.seasonal_lookback_frequency.setEnabled(False)

        layout.addWidget(QLabel("Train data size:"), 0, 0)
        layout.addWidget(self.train_data_size, 0, 1)

        layout.addWidget(QLabel("Scale type:"), 1, 0)
        layout.addWidget(self.scale_type, 1, 1)

        layout.addWidget(self.lookback_option, 2, 0, 1, 2, Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(QLabel("Value:"), 3, 0)
        layout.addWidget(self.lookback_value, 3, 1)

        layout.addWidget(self.seasonal_lookback_option, 4, 0, 1, 2, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(QLabel("Value:"), 5, 0)
        layout.addWidget(self.seasonal_lookback_value, 5, 1)
        layout.addWidget(QLabel("Frequency:"), 6, 0)
        layout.addWidget(self.seasonal_lookback_frequency, 6, 1)

        self.lookback_option.stateChanged.connect(self.update_lookback_state)
        self.seasonal_lookback_option.stateChanged.connect(self.update_seasonal_lookback_state)

        # Connect all input widgets to emit paramsChanged signal
        self.train_data_size.textChanged.connect(self.paramsChanged.emit)
        self.scale_type.currentIndexChanged.connect(self.paramsChanged.emit)
        self.lookback_option.stateChanged.connect(self.paramsChanged.emit)
        self.lookback_value.textChanged.connect(self.paramsChanged.emit)
        self.seasonal_lookback_option.stateChanged.connect(self.paramsChanged.emit)
        self.seasonal_lookback_value.textChanged.connect(self.paramsChanged.emit)
        self.seasonal_lookback_frequency.textChanged.connect(self.paramsChanged.emit)

    def update_lookback_state(self, state):
        self.lookback_value.setEnabled(state == Qt.CheckState.Checked)

    def update_seasonal_lookback_state(self, state):
        self.seasonal_lookback_value.setEnabled(state == Qt.CheckState.Checked)
        self.seasonal_lookback_frequency.setEnabled(state == Qt.CheckState.Checked)

    def get_params(self) -> Dict[str, Any]:
        return {
            "train_data_size": int(self.train_data_size.text()),
            "scale_type": self.scale_type.currentText(),
            "lookback_option": self.lookback_option.isChecked(),
            "lookback_value": int(self.lookback_value.text()) if self.lookback_value.text() else None,
            "seasonal_lookback_option": self.seasonal_lookback_option.isChecked(),
            "seasonal_lookback_value": int(self.seasonal_lookback_value.text()) if self.seasonal_lookback_value.text() else None,
            "seasonal_lookback_frequency": int(self.seasonal_lookback_frequency.text()) if self.seasonal_lookback_frequency.text() else None,
        }

    def set_params(self, data: Dict[str, Any]) -> None:
        self.train_data_size.setText(str(data["train_data_size"]))
        self.scale_type.setCurrentText(data["scale_type"])
        self.lookback_option.setChecked(data["lookback_option"])
        self.lookback_value.setText(str(data["lookback_value"]) if data["lookback_option"] else "")
        self.seasonal_lookback_option.setChecked(data["seasonal_lookback_option"])
        self.seasonal_lookback_value.setText(str(data["seasonal_lookback_value"]) if data["seasonal_lookback_option"] else "")
        self.seasonal_lookback_frequency.setText(str(data["seasonal_lookback_frequency"]) if data["seasonal_lookback_option"] else "")

    def check_errors(self) -> None:
        train_data_size = int(self.train_data_size.text())
        if train_data_size < 0:
            raise ValueError("Train data size must be greater than 0")
        if train_data_size > 100:
            raise ValueError("Train data size must be less than 100")

        if self.lookback_option.isChecked() and not self.lookback_value.text():
            raise ValueError("Lookback value must be specified")

        if self.seasonal_lookback_option.isChecked():
            if not self.seasonal_lookback_value.text():
                raise ValueError("Seasonal lookback value must be specified")
            if not self.seasonal_lookback_frequency.text():
                raise ValueError("Seasonal lookback frequency must be specified")

    def grid(self, layout: QGridLayout, column: int, row: int) -> "PreprocessingComponent":
        layout.addWidget(self.root, row, column)
        return self
