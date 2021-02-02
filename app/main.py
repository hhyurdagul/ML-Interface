import tkinter as tk
from tkinter import ttk

import warnings
warnings.filterwarnings("ignore")

from gui.timeseries import TimeSeries
from gui.supportvectormachine import SupportVectorMachine
from gui.generalregression import GeneralRegressionNeuralNetwork
from gui.mlp import MultiLayerPerceptron
from gui.sarima import SARIMA
from gui.montecarlo import MonteCarlo
from gui.movingaverage import MovingAverage

class GUI:
    def __init__(self):
        self.gui = tk.Tk()
        self.parent = ttk.Notebook(self.gui)

        time_series = TimeSeries()
        self.add(time_series, "Time Series")
        
        svm = SupportVectorMachine()
        self.add(svm, "Support Vector Machine")
      
        grnn = GeneralRegressionNeuralNetwork()
        self.add(grnn, "GRNN")

        mlp = MultiLayerPerceptron()
        self.add(mlp, "Mlp")
        
        sarima = SARIMA()
        self.add(sarima, "SARIMA")
        
        monte = MonteCarlo()
        self.add(monte, "Monte Carlo")
        
        moving_average = MovingAverage()
        self.add(moving_average, "Moving Average")

        self.parent.pack(expand=1, fill='both')

    def add(self, frame, text):
        self.parent.add(frame.root, text=text)

    def start(self):
        self.gui.mainloop()

s = GUI()
s.start()
