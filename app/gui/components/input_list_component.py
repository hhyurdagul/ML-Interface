import tkinter as tk
from tkinter import ttk, filedialog
from typing import Union, List, Any
from .utils import popupmsg
from ..backend import DataHandler


class InputListComponent:
    def __init__(self, root: ttk.Frame, data_handler: DataHandler) -> None:
        self.data_handler = data_handler

        frame_root = ttk.Labelframe(root, text="Get Train Set")
        frame_root.grid(column=0, row=0)

        file_path = tk.StringVar(value="")
        ttk.Label(frame_root, text="Train File Path").grid(column=0, row=0)
        ttk.Entry(frame_root, textvariable=file_path).grid(column=1, row=0)
        ttk.Button(
            frame_root,
            text="Read Data",
            command=lambda: self.read_train_data(file_path),
        ).grid(column=2, row=0)

        self.input_list = tk.Listbox(frame_root)
        self.input_list.grid(column=0, row=1)
        self.input_list.bind("<Double-Button-1>", self.add_predictor)
        self.input_list.bind("<Double-Button-3>", self.add_target)

        self.predictor_list = tk.Listbox(frame_root)
        self.predictor_list.grid(column=1, row=1)
        self.predictor_list.bind("<Double-Button-1>", self.eject_predictor)

        self.target_list = tk.Listbox(frame_root)
        self.target_list.grid(column=2, row=1)
        self.target_list.bind("<Double-Button-1>", self.eject_target)

        ttk.Button(frame_root, text="Add Predictor", command=self.add_predictor).grid(
            column=1, row=2
        )
        ttk.Button(
            frame_root, text="Eject Predictor", command=self.eject_predictor
        ).grid(column=1, row=3)

        ttk.Button(frame_root, text="Add Target", command=self.add_target).grid(
            column=2, row=2
        )
        ttk.Button(frame_root, text="Eject Target", command=self.eject_target).grid(
            column=2, row=3
        )

    def check_errors(self) -> bool:
        predictor_names = self.get_predictor_names()
        target_name = self.get_target_name()
        if len(predictor_names) == 0:
            return popupmsg("Select predictors")

        elif target_name == "":
            return popupmsg("Select a target")

        elif target_name in predictor_names:
            return popupmsg("Target and predictor have same variable")

        if not self.data_handler.train_df_read:
            return popupmsg("Read a data first")

        return True

    def get_params(self) -> dict[str, Union[list[str], str]]:
        return {
            "predictor_names": self.get_predictor_names(),
            "label_name": self.get_target_name(),
            "is_round": self.data_handler.is_round,
            "is_negative": self.data_handler.is_negative,
        }

    def set_params(self, params: dict[str, Any]):
        self.data_handler.predictor_names = params.get("predictor_names", [])
        self.data_handler.label_name = params.get("label_name", "")

        self.data_handler.is_round = params.get("is_round", True)
        self.data_handler.is_negative = params.get("is_negative", False)

        names = "\n".join(self.data_handler.predictor_names)
        msg = (
            f"Predictor names are {names}\nLabel name is {self.data_handler.label_name}"
        )
        popupmsg(msg)

    def get_predictor_names(self) -> List[str]:
        return list(self.predictor_list.get(0, tk.END))

    def get_target_name(self) -> str:
        return self.target_list.get(0)

    def add_predictor(self, _=None) -> None:
        selected = self.input_list.curselection()
        if selected != "":
            item = self.input_list.get(selected)
            if item not in self.predictor_list.get(0, tk.END):
                self.predictor_list.insert(tk.END, item)

    def eject_predictor(self, _=None) -> None:
        selected = self.predictor_list.curselection()
        if selected != "":
            self.predictor_list.delete(selected)

    def add_target(self, _=None) -> None:
        selected = self.input_list.curselection()
        if selected != "" and self.target_list.size() < 1:
            item = self.input_list.get(selected)
            self.target_list.insert(tk.END, item)

    def eject_target(self, _=None) -> None:
        selected = self.target_list.curselection()
        if selected != "":
            self.target_list.delete(selected)

    def read_train_data(self, file_path: tk.StringVar) -> None:
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

        columns = self.data_handler.read_train_data(path)

        self.input_list.delete(0, tk.END)
        self.predictor_list.delete(0, tk.END)
        self.target_list.delete(0, tk.END)

        for column in columns:
            self.input_list.insert(tk.END, column)
