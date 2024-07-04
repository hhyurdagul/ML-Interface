import tkinter as tk
from tkinter import filedialog, ttk
from typing import Callable


class InputComponent:
    def __init__(
        self,
        parent: ttk.Frame,
        text: str,
        read_func: Callable[[str], list[str]] = lambda _: list(),
    ):
        self.root = ttk.LabelFrame(parent, text=text)
        self.read_func = read_func

        self.file_path = tk.StringVar(value="")
        ttk.Label(self.root, text="Train File Path").grid(row=0, column=0)
        ttk.Entry(self.root, textvariable=self.file_path).grid(row=0, column=1)

        ttk.Button(
            self.root,
            text="Read Data",
            command=self.read_train_data,
        ).grid(column=2, row=0)

        self.input_list = tk.Listbox(self.root, selectmode=tk.EXTENDED)
        self.input_list.grid(column=0, row=1)

        self.predictor_list = tk.Listbox(self.root, selectmode=tk.EXTENDED)
        self.predictor_list.grid(column=1, row=1)

        self.target_list = tk.Listbox(self.root)
        self.target_list.grid(column=2, row=1)

        ttk.Button(self.root, text="Add Predictor", command=self.add_predictor).grid(
            column=1, row=2
        )
        ttk.Button(self.root, text="Eject Predictor", command=self.eject_predictor).grid(
            column=1, row=3
        )

        ttk.Button(self.root, text="Add Target", command=self.add_target).grid(
            column=2, row=2
        )
        ttk.Button(self.root, text="Eject Target", command=self.eject_target).grid(
            column=2, row=3
        )
    
    def __fill_input_list(self, columns: list[str]) -> None:
        self.input_list.delete(0, tk.END)
        self.input_list.insert(tk.END, *columns)
        self.predictor_list.delete(0, tk.END)
        self.target_list.delete(0, tk.END)

    def read_train_data(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[
                ("Xlsx Files", "*.xlsx"),
                ("Xlrd Files", "*.xls"),
                ("Csv Files", "*.csv"),
            ]
        )
        if not path:
            return
        self.file_path.set(path)

        columns = self.read_func(path)
        self.__fill_input_list(columns)

    def add_predictor(self) -> None:
        selected = self.input_list.curselection()
        for i in selected:
            item = self.input_list.get(i)
            if item in self.predictor_list.get(0, tk.END):
                continue
            self.predictor_list.insert(tk.END, self.input_list.get(i))

    def eject_predictor(self) -> None:
        for i in self.predictor_list.curselection():
            self.predictor_list.delete(i)

    def add_target(self) -> None:
        selected = self.input_list.curselection()
        if selected and self.target_list.size() < 1:
            self.target_list.insert(tk.END, self.input_list.get(selected[0]))

    def eject_target(self) -> None:
        selected = self.target_list.curselection()
        if selected:
            self.target_list.delete(selected)

    def get_predictors(self) -> list[str]:
        return list(self.predictor_list.get(0, tk.END))

    def get_target(self) -> str:
        return self.target_list.get(0)

    def check_errors(self) -> None:
        if self.input_list.size() < 1:
            raise Exception("Read a data first")
        if self.predictor_list.size() < 1:
            raise Exception("Predictor list is empty")
        if self.target_list.size() < 1:
            raise Exception("Target list is empty")
        if self.get_target() in self.get_predictors():
            raise Exception("Target and predictor cannot be the same")

    def grid(self, column: int, row: int) -> "InputComponent":
        self.root.grid(column=column, row=row)
        return self
