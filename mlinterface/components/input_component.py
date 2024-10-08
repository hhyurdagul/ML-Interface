from PySide6.QtWidgets import (QWidget, QLabel, QLineEdit, QPushButton, QListWidget,
                               QGridLayout, QFileDialog, QGroupBox, QAbstractItemView)
from PySide6.QtCore import Signal, Qt
from typing import Callable

class InputComponent(QGroupBox):
    def __init__(
        self,
        title: str,
        read_func: Callable[[str], list[str]] = lambda _: list(),
    ):
        super().__init__(title)
        self.__read_func = read_func
        layout = QGridLayout()

        self.__file_path = QLineEdit()
        self.__file_path.setFixedWidth(150)
        self.__file_path.setEnabled(False)
        layout.addWidget(QLabel("Train File Path:"), 0, 0)
        layout.addWidget(self.__file_path, 0, 1)

        read_button = QPushButton("Read Data")
        read_button.clicked.connect(self.__read_train_data)
        layout.addWidget(read_button, 0, 2)

        self.__input_list = QListWidget()
        self.__input_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.__input_list.setFixedWidth(150)
        layout.addWidget(self.__input_list, 1, 0)

        self.__predictor_list = QListWidget()
        self.__predictor_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.__predictor_list.setFixedWidth(150)
        layout.addWidget(self.__predictor_list, 1, 1)

        self.__target_list = QListWidget()
        self.__target_list.setFixedWidth(150)
        layout.addWidget(self.__target_list, 1, 2)

        add_predictor_button = QPushButton("Add Predictor")
        add_predictor_button.clicked.connect(self.__add_predictor)
        layout.addWidget(add_predictor_button, 2, 1)

        eject_predictor_button = QPushButton("Eject Predictor")
        eject_predictor_button.clicked.connect(self.__eject_predictor)
        layout.addWidget(eject_predictor_button, 3, 1)

        add_target_button = QPushButton("Add Target")
        add_target_button.clicked.connect(self.__add_target)
        layout.addWidget(add_target_button, 2, 2)

        eject_target_button = QPushButton("Eject Target")
        eject_target_button.clicked.connect(self.__eject_target)
        layout.addWidget(eject_target_button, 3, 2)

        self.setLayout(layout)

    def __fill_input_list(self, values: list[str]) -> None:
        self.__input_list.clear()
        self.__input_list.addItems(values)
        self.__predictor_list.clear()
        self.__target_list.clear()

    def __read_train_data(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open File",
            "",
            "Xlsx Files (*.xlsx);;Xlrd Files (*.xls);;Csv Files (*.csv)"
        )
        if not path:
            return
        self.__file_path.setText(path)

        data_columns = self.__read_func(path)
        self.__fill_input_list(data_columns)

    def __add_predictor(self) -> None:
        for item in self.__input_list.selectedItems():
            if self.__predictor_list.findItems(item.text(), Qt.MatchFlag.MatchExactly):
                continue
            self.__predictor_list.addItem(item.text())

    def __eject_predictor(self) -> None:
        for item in self.__predictor_list.selectedItems():
            self.__predictor_list.takeItem(self.__predictor_list.row(item))

    def __add_target(self) -> None:
        selected = self.__input_list.selectedItems()
        if selected and self.__target_list.count() < 1:
            self.__target_list.addItem(selected[0].text())

    def __eject_target(self) -> None:
        self.__target_list.clear()

    def get_predictors(self) -> list[str]:
        return [self.__predictor_list.item(i).text() for i in range(self.__predictor_list.count())]

    def get_target(self) -> str:
        return self.__target_list.item(0).text() if self.__target_list.count() > 0 else ""

    def check_errors(self) -> None:
        if self.__input_list.count() < 1:
            raise Exception("Read a data first")
        if self.__predictor_list.count() < 1:
            raise Exception("Predictor list is empty")
        if self.__target_list.count() < 1:
            raise Exception("Target list is empty")
        if self.get_target() in self.get_predictors():
            raise Exception("Target and predictor cannot be the same")
