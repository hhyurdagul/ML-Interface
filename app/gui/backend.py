import json
import tkinter as tk
from tkinter import ttk


class BaseSupervisedModel:
    def __init__(self):
        self.root = ttk.Frame()

        # Get Train Set
        get_train_set_frame = ttk.Labelframe(self.root, text="Get Train Set")
        get_train_set_frame.grid(column=0, row=0)

        file_path = tk.StringVar(value="")
        ttk.Label(get_train_set_frame, text="Train File Path").grid(column=0, row=0)
        ttk.Entry(get_train_set_frame, textvariable=file_path).grid(column=1, row=0)
        ttk.Button(
            get_train_set_frame,
            text="Read Data",
            command=lambda: self.read_train_data(file_path),
        ).grid(column=2, row=0)

        self.input_list = tk.Listbox(get_train_set_frame)
        self.input_list.grid(column=0, row=1)
        self.input_list.bind("<Double-Button-1>", self.add_predictor)
        self.input_list.bind("<Double-Button-3>", self.add_target)

        self.predictor_list = tk.Listbox(get_train_set_frame)
        self.predictor_list.grid(column=1, row=1)
        self.predictor_list.bind("<Double-Button-1>", self.eject_predictor)

        self.target_list = tk.Listbox(get_train_set_frame)
        self.target_list.grid(column=2, row=1)
        self.target_list.bind("<Double-Button-1>", self.eject_target)

        ttk.Button(
            get_train_set_frame, text="Add Predictor", command=self.add_predictor
        ).grid(column=1, row=2)
        ttk.Button(
            get_train_set_frame, text="Eject Predictor", command=self.eject_predictor
        ).grid(column=1, row=3)

        ttk.Button(
            get_train_set_frame, text="Add Target", command=self.add_target
        ).grid(column=2, row=2)
        ttk.Button(
            get_train_set_frame, text="Eject Target", command=self.eject_target
        ).grid(column=2, row=3)
