import tkinter as tk
from tkinter import filedialog, ttk
from typing import Callable

class DataTable(tk.Frame):
    def __init__(
        self,
        parent: tk.Tk,
        data: list[tuple[float, float]],
        save_func: Callable = lambda: None,
    ):
        tk.Frame.__init__(self, parent)
        self.data = data
        self.save_func = save_func
        self.table = ttk.Treeview(
            parent, columns=("idx", "real", "predict"), show="headings"
        )
        self.table.heading("idx")
        self.table.column("idx", minwidth=10, width=30, stretch=False, anchor=tk.CENTER)
        self.table.heading("real", text="Real")
        self.table.column("real", anchor=tk.CENTER)
        self.table.heading("predict", text="Predict")
        self.table.column("predict", anchor=tk.CENTER)

        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")

        self.table.pack(expand=True, fill="both")

        button = tk.Button(parent, text="Save data", command=self.save_data)
        button.pack()

        self.update_data()

    def update_data(self) -> None:
        self.table.delete(*self.table.get_children())
        for idx, row in enumerate(self.data):
            self.table.insert("", tk.END, values=(str(idx + 1),) + row)

    def save_data(self) -> None:
        file_path = filedialog.asksaveasfilename(
            filetypes=[("Excel File", "*.xlsx"), ("CSV File", "*.csv")]
        )
        if file_path:
            self.save_func(file_path)

        return None

