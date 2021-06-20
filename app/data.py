import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

import numpy as np
import pandas as pd

import pandastable

holiday_dates = [
                 {"month":1, "day":1},
                 {"month":4, "day":23},
                 {"month":5, "day":1},
                 {"month":5, "day":19},
                 {"month":7, "day":15},
                 {"month":8, "day":30},
                 {"month":10, "day":29}
]


def isWeekend(d):
    return 1 if d.weekday() == 5 or d.weekday() == 6 else 0

class Main:
    def __init__(self):
        self.root = tk.Tk()
        
        first_frame = ttk.Frame(self.root)
        first_frame.grid(column=0, row=0)

        path = tk.Variable(value="")
        ttk.Entry(first_frame, textvariable=path).grid(column=0, row=0)
        ttk.Button(first_frame, text="Get Data", command=lambda: self.getData(path)).grid(column=1, row=0)

        self.input_list = tk.Listbox(first_frame)
        self.input_list.grid(column=0, row=1, columnspan=2)

        self.date_var = tk.Variable(value="Select the date column")
        ttk.Button(first_frame, textvariable=self.date_var, command=self.selectDate).grid(column=0, row=2, columnspan=2)

        ttk.Label(first_frame, text="Select Variables to Add").grid(column=0, row=3, columnspan=2)

        self.choices = [tk.IntVar(value=0), tk.IntVar(value=0), tk.IntVar(value=0), tk.IntVar(value=0), tk.IntVar(value=0), tk.IntVar(value=0), tk.IntVar(value=0)]
        names = ["Minute", "Hour", "Day", "Month", "Year", "Weekend", "Holiday"]

        for i,j in enumerate(names):
            tk.Checkbutton(first_frame, text=j, offvalue=0, onvalue=1, variable=self.choices[i]).grid(column=0, row=i+4, sticky=tk.W)

        ttk.Button(first_frame, text="Modify Dataset", command=self.modifyDataset, width=12).grid(column=1, row=4, rowspan=2)
        ttk.Button(first_frame, text="Show Dataset", width=12, command=self.showDataset).grid(column=1, row=6, rowspan=2)
        ttk.Button(first_frame, text="Save Dataset", width=12, command=lambda: self.saveFile(path.get())).grid(column=1, row=8, rowspan=2)

    def getData(self, filepath):
        path = filedialog.askopenfilename(filetypes=[("All Files", "*")])
        filepath.set(path)
        if path.endswith("xlsx") or path.endswith("xls"):
            self.df = pd.read_excel(path)
        else:
            self.df = pd.read_csv(path)
        
        self.input_list.delete(0, tk.END)
        for i in self.df.columns:
            self.input_list.insert(tk.END, i)

    def selectDate(self):
        a = self.input_list.get(self.input_list.curselection())
        self.date_var.set("Selected column " + a)
        self.date = pd.to_datetime(self.df[a])

    def options(self, num):
        date = self.date
        if num == 0:
            self.df["Minute"] = date.apply(lambda x: x.minute)
        elif num == 1:
            self.df["Hour"] = date.apply(lambda x: x.hour)
        elif num == 2:
            self.df["Day"] = date.apply(lambda x: x.day)
        elif num == 3:
            self.df["Month"] = date.apply(lambda x: x.month)
        elif num == 4:
            self.df["Year"] = date.apply(lambda x: x.year)
        elif num == 5:
            self.df["Weekend"] = date.apply(isWeekend)
        elif num == 6:
            self.df["Holiday"] = date.apply(isHoliday)

    def modifyDataset(self):
        for i,j in enumerate(self.choices):
            if j.get() == 1:
                self.options(i)

    def showDataset(self):
        top = tk.Toplevel(self.root)
        pt = pandastable.Table(top, dataframe=self.df, editable=False)
        pt.show()

    def saveFile(self, path):
        if(path.endswith(".csv")):
            self.df.to_csv(path, index=False)
        else:
            self.df.to_excel(path, index=False)

g = Main()
g.root.mainloop()
