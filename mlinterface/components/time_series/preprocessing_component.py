from PySide6.QtWidgets import (QWidget, QLabel, QLineEdit, QComboBox, QCheckBox,
                               QGridLayout, QGroupBox)
from PySide6.QtCore import Qt
from typing import Any, Dict

from mlinterface.components.variables import QIntLineEdit


class PreprocessingComponent(QGroupBox):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.setTitle(title)
        layout = QGridLayout()

        self.train_data_size = QIntLineEdit("100", 1, 100)
        self.train_data_size.setFixedWidth(100)

        self.scale_type = QComboBox()
        self.scale_type.addItems(["None", "MinMax", "Standard"])

        self.lookback_option = QCheckBox("Lookback")
        self.lookback_value = QIntLineEdit("", 1, 1000)
        self.lookback_value.setEnabled(False)

        self.seasonal_lookback_option = QCheckBox("Periodic Lookback")
        self.seasonal_lookback_value = QIntLineEdit("", 1, 1000)
        self.seasonal_lookback_value.setEnabled(False)

        self.seasonal_lookback_frequency = QIntLineEdit("", 1, 1000)
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

        self.lookback_option.stateChanged.connect(
            lambda state: self.__lookback_state_change(state)
        )
        self.seasonal_lookback_option.stateChanged.connect(
            lambda state: self.__seasonal_lookback_state_change(state)
        )
        self.setLayout(layout)

    def __lookback_state_change(self, state: int) -> None:
        print("State", state)
        if state == 2:
            self.lookback_value.setEnabled(True)
            self.lookback_value.setText("1")
        else:
            self.lookback_value.setEnabled(False)
            self.lookback_value.setText("")

    def __seasonal_lookback_state_change(self, state: int) -> None:
        if state == 2:
            self.seasonal_lookback_value.setEnabled(True)
            self.seasonal_lookback_frequency.setEnabled(True)
            self.seasonal_lookback_value.setText("1")
            self.seasonal_lookback_frequency.setText("1")
        else:
            self.seasonal_lookback_value.setEnabled(False)
            self.seasonal_lookback_frequency.setEnabled(False)
            self.seasonal_lookback_value.setText("")
            self.seasonal_lookback_frequency.setText("")

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
