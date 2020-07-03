import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

class GUI:
    def __init__(self):
        self.root = tk.Tk()
        
        # Get Train Set
        get_train_set_frame = ttk.LabelFrame(self.root, text="Get Train Set")
        get_train_set_frame.grid(column=0, row=0)

        file_path = tk.StringVar(value="")
        ttk.Label(get_train_set_frame, text="Train File Path").grid(column=0, row=0)
        ttk.Entry(get_train_set_frame, textvariable=file_path).grid(column=1, row=0)
        ttk.Button(get_train_set_frame, text="Read Csv", command=lambda: self.readCsv(file_path)).grid(column=2, row=0)

        self.input_list = tk.Listbox(get_train_set_frame)
        self.input_list.grid(column=0, row=1)

        self.predictor_list = tk.Listbox(get_train_set_frame)
        self.predictor_list.grid(column=1, row=1)

        self.target_list = tk.Listbox(get_train_set_frame)
        self.target_list.grid(column=2, row=1)

        ttk.Button(get_train_set_frame, text="Add Predictor", command=self.addPredictor).grid(column=1, row=2)
        ttk.Button(get_train_set_frame, text="Eject Predictor", command=self.ejectPredictor).grid(column=1, row=3)

        ttk.Button(get_train_set_frame, text="Add Target", command=self.addTarget).grid(column=2, row=2)
        ttk.Button(get_train_set_frame, text="Eject Target", command=self.ejectTarget).grid(column=2, row=3)
       
        # Customize Train Set
        customize_train_set_frame = tk.LabelFrame(self.root, text="Customize Train Set")
        customize_train_set_frame.grid(column=0, row=1)

        self.train_size_var = tk.IntVar(value="")
        ttk.Label(customize_train_set_frame, text="# of Rows in Train Set").grid(column=0, row=0)
        ttk.Entry(customize_train_set_frame, textvariable=self.train_size_var).grid(column=1, row=0)

        self.size_choice_var = tk.IntVar(value=0)
        tk.Radiobutton(customize_train_set_frame, text="As Percent", value=0, variable=self.size_choice_var).grid(column=0, row=1)
        tk.Radiobutton(customize_train_set_frame, text="As Number", value=1, variable=self.size_choice_var).grid(column=1, row=1)

        self.scale_var = tk.StringVar(value="None")
        ttk.Label(customize_train_set_frame, text="Scale Type").grid(column=0, row=2)
        ttk.OptionMenu(customize_train_set_frame, self.scale_var, "None", "None","StandardScaler", "MinMaxScaler").grid(column=1, row=2)

        # Lag Options
        lag_options_frame = ttk.LabelFrame(self.root, text="Lag Options")
        lag_options_frame.grid(column=0, row=2)

        acf_lags = tk.IntVar(value="")
        ttk.Label(lag_options_frame, text="Number of Lags").grid(column=0, row=0)
        ttk.Entry(lag_options_frame, textvariable=acf_lags).grid(column=1, row=0)
        ttk.Button(lag_options_frame, text="Show ACF", command=lambda: self.showACF(acf_lags.get())).grid(column=2, row=0)

        self.lag_option_var = tk.IntVar(value="")
        tk.Radiobutton(lag_options_frame, text="Use All Lags", value=0, variable=self.lag_option_var, command=self.openEntries).grid(column=0, row=1)
        tk.Radiobutton(lag_options_frame, text="Use Selected(1,3,..)", value=1, variable=self.lag_option_var, command=self.openEntries).grid(column=0, row=2)
        tk.Radiobutton(lag_options_frame, text="Use Best N", value=2, variable=self.lag_option_var, command=self.openEntries).grid(column=0, row=3)
        tk.Radiobutton(lag_options_frame, text="Use Correlation > n", value=3, variable=self.lag_option_var, command=self.openEntries).grid(column=0, row=4)
        
        self.lag_entries = [ttk.Entry(lag_options_frame, state=tk.DISABLED) for i in range(3)]
        [self.lag_entries[i-2].grid(column=1, row=i) for i in range(2,5)]

               

    def readCsv(self, file_path):
        path = filedialog.askopenfilename(filetypes=[("Csv Files", "*.csv")])
        file_path.set(path)
        self.df = pd.read_csv(path)
        
        self.input_list.delete(0, tk.END)

        for i in self.df.columns:
            self.input_list.insert(tk.END, i)

    def addPredictor(self):
        try:
            a = self.input_list.get(self.input_list.curselection())
            if a not in self.predictor_list.get(0,tk.END):
                self.predictor_list.insert(tk.END, a)
        except:
            pass

    def ejectPredictor(self):
        try:
            a = self.predictor_list.delete(self.predictor_list.curselection())
        except:
            pass
    
    def addTarget(self):
        try:
            a = self.input_list.get(self.input_list.curselection())
            if self.target_list.size() < 1:
                self.target_list.insert(tk.END, a)
        except:
            pass

    def ejectTarget(self):
        try:
            a = self.target_list.delete(self.target_list.curselection())
        except:
            pass

    def showACF(self, lags):
        top = tk.Toplevel()
        fig = plt.Figure((20, 15))
        
        data = self.df[self.target_list.get(0)]

        ax = fig.add_subplot(211)
        plot_acf(data, ax=ax, lags=lags)

        ax1 = fig.add_subplot(212)
        plot_pacf(data, ax=ax1, lags=lags)

        canvas = FigureCanvasTkAgg(fig, top)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, top)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def openEntries(self):
        o = self.lag_option_var.get() - 1
        for i, j in enumerate(self.lag_entries):
            if i == o:
                j["state"] = tk.NORMAL
            else:
                j["state"] = tk.DISABLED

    def start(self):
        self.root.mainloop()



s = GUI()
s.start()

