# Gui
import tkinter as tk
from tkinter import ttk

from gui.timeseries import TimeSeries
from gui.supportvectormachine import SupportVectorMachine
from gui.generalregression import GeneralizedRegressionNeuralNetwork

class GUI:
    def __init__(self):
        self.gui = tk.Tk()
        self.parent = ttk.Notebook(self.gui)

        time_series = TimeSeries()
        self.add(time_series, "Time Series")
        
        svm = SupportVectorMachine()
        self.add(svm, "Support Vector Machine")
      
        grnn = GeneralizedRegressionNeuralNetwork()
        self.add(grnn, "GRNN")

        self.parent.pack(expand=1, fill='both')

    def add(self, frame, text):
        self.parent.add(frame.root, text=text)

    def start(self):
        self.gui.mainloop()


s = GUI()
s.start()

