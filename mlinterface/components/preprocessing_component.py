from PySide6.QtWidgets import (QWidget, QLabel, QLineEdit, QComboBox, QCheckBox,
                               QGridLayout, QGroupBox)
from PySide6.QtCore import Qt
from typing import Any, Dict

from mlinterface.components.variables import QIntLineEdit


class PreprocessingComponent(QGroupBox):
    def __init__(self, title: str) -> None:
        super().__init__(title)
        layout = QGridLayout()

        self.__train_data_size = QIntLineEdit("100", 1, 100)
        self.__train_data_size.setFixedWidth(100)

        self.__scale_type = QComboBox()
        self.__scale_type.setFixedWidth(100)
        self.__scale_type.addItems(["None", "MinMax", "Standard"])

        self.__lookback_option = QCheckBox("Lookback")
        self.__lookback_value = QIntLineEdit("", 1, 1000)
        self.__lookback_value.setFixedWidth(100)
        self.__lookback_value.setEnabled(False)

        self.__seasonal_lookback_option = QCheckBox("Periodic Lookback")
        self.__seasonal_lookback_value = QIntLineEdit("", 1, 1000)
        self.__seasonal_lookback_value.setFixedWidth(100)
        self.__seasonal_lookback_value.setEnabled(False)

        self.__seasonal_lookback_frequency = QIntLineEdit("", 1, 1000)
        self.__seasonal_lookback_frequency.setFixedWidth(100)
        self.__seasonal_lookback_frequency.setEnabled(False)

        layout.addWidget(QLabel("Train data size:"), 0, 0)
        layout.addWidget(self.__train_data_size, 0, 1)

        layout.addWidget(QLabel("Scale type:"), 1, 0)
        layout.addWidget(self.__scale_type, 1, 1)

        layout.addWidget(self.__lookback_option, 2, 0, Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(QLabel("Value:"), 3, 0)
        layout.addWidget(self.__lookback_value, 3, 1)

        layout.addWidget(self.__seasonal_lookback_option, 4, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(QLabel("Value:"), 5, 0)
        layout.addWidget(self.__seasonal_lookback_value, 5, 1)
        layout.addWidget(QLabel("Frequency:"), 6, 0)
        layout.addWidget(self.__seasonal_lookback_frequency, 6, 1)

        self.__lookback_option.stateChanged.connect(
            lambda state: self.__lookback_state_change(state)
        )
        self.__seasonal_lookback_option.stateChanged.connect(
            lambda state: self.__seasonal_lookback_state_change(state)
        )
        self.setLayout(layout)

    def __lookback_state_change(self, state: int) -> None:
        print("State", state)
        if state == 2:
            self.__lookback_value.setEnabled(True)
            self.__lookback_value.setText("1")
        else:
            self.__lookback_value.setEnabled(False)
            self.__lookback_value.setText("")

    def __seasonal_lookback_state_change(self, state: int) -> None:
        if state == 2:
            self.__seasonal_lookback_value.setEnabled(True)
            self.__seasonal_lookback_frequency.setEnabled(True)
            self.__seasonal_lookback_value.setText("1")
            self.__seasonal_lookback_frequency.setText("1")
        else:
            self.__seasonal_lookback_value.setEnabled(False)
            self.__seasonal_lookback_frequency.setEnabled(False)
            self.__seasonal_lookback_value.setText("")
            self.__seasonal_lookback_frequency.setText("")

    def get_params(self) -> Dict[str, Any]:
        return {
            "train_data_size": int(self.__train_data_size.text()),
            "scale_type": self.__scale_type.currentText(),
            "lookback_option": self.__lookback_option.isChecked(),
            "lookback_value": int(self.__lookback_value.text()) if self.__lookback_value.text() else None,
            "seasonal_lookback_option": self.__seasonal_lookback_option.isChecked(),
            "seasonal_lookback_value": int(self.__seasonal_lookback_value.text()) if self.__seasonal_lookback_value.text() else None,
            "seasonal_lookback_frequency": int(self.__seasonal_lookback_frequency.text()) if self.__seasonal_lookback_frequency.text() else None,
        }

    def set_params(self, data: Dict[str, Any]) -> None:
        self.__train_data_size.setText(str(data["train_data_size"]))
        self.__scale_type.setCurrentText(data["scale_type"])
        self.__lookback_option.setChecked(data["lookback_option"])
        self.__lookback_value.setText(str(data["lookback_value"]) if data["lookback_option"] else "")
        self.__seasonal_lookback_option.setChecked(data["seasonal_lookback_option"])
        self.__seasonal_lookback_value.setText(str(data["seasonal_lookback_value"]) if data["seasonal_lookback_option"] else "")
        self.__seasonal_lookback_frequency.setText(str(data["seasonal_lookback_frequency"]) if data["seasonal_lookback_option"] else "")
