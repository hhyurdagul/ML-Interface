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
        self.read_func = read_func
        layout = QGridLayout()

        self.file_path = QLineEdit()
        self.file_path.setFixedWidth(150)
        layout.addWidget(QLabel("Train File Path:"), 0, 0)
        layout.addWidget(self.file_path, 0, 1)

        read_button = QPushButton("Read Data")
        read_button.clicked.connect(self.__read_train_data)
        layout.addWidget(read_button, 0, 2)

        self.input_list = QListWidget()
        self.input_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.input_list.setFixedWidth(150)
        layout.addWidget(self.input_list, 1, 0)

        self.predictor_list = QListWidget()
        self.predictor_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.predictor_list.setFixedWidth(150)
        layout.addWidget(self.predictor_list, 1, 1)

        self.target_list = QListWidget()
        self.target_list.setFixedWidth(150)
        layout.addWidget(self.target_list, 1, 2)

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
        self.input_list.clear()
        self.input_list.addItems(values)
        self.predictor_list.clear()
        self.target_list.clear()

    def __read_train_data(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open File",
            "",
            "Xlsx Files (*.xlsx);;Xlrd Files (*.xls);;Csv Files (*.csv)"
        )
        if not path:
            return
        self.file_path.setText(path)

        data_columns = self.read_func(path)
        self.__fill_input_list(data_columns)

    def __add_predictor(self) -> None:
        for item in self.input_list.selectedItems():
            if self.predictor_list.findItems(item.text(), Qt.MatchFlag.MatchExactly):
                continue
            self.predictor_list.addItem(item.text())

    def __eject_predictor(self) -> None:
        for item in self.predictor_list.selectedItems():
            self.predictor_list.takeItem(self.predictor_list.row(item))

    def __add_target(self) -> None:
        selected = self.input_list.selectedItems()
        if selected and self.target_list.count() < 1:
            self.target_list.addItem(selected[0].text())

    def __eject_target(self) -> None:
        self.target_list.clear()

    def get_predictors(self) -> list[str]:
        return [self.predictor_list.item(i).text() for i in range(self.predictor_list.count())]

    def get_target(self) -> str:
        return self.target_list.item(0).text() if self.target_list.count() > 0 else ""

    def check_errors(self) -> None:
        if self.input_list.count() < 1:
            raise Exception("Read a data first")
        if self.predictor_list.count() < 1:
            raise Exception("Predictor list is empty")
        if self.target_list.count() < 1:
            raise Exception("Target list is empty")
        if self.get_target() in self.get_predictors():
            raise Exception("Target and predictor cannot be the same")
