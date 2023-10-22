# from gui.test import Test
from gui.timeseries import TimeSeries
from gui.mlp import MultiLayerPerceptron
from gui.supportvectormachine import SupportVectorMachine
from gui.random_forest import RandomForest
from gui.lgbm import LGBM
from gui.catboost_arch import CatBoost
from gui.xg import XGB
from gui.ridge import Ridge
from gui.linear_regression import LinearModel
from gui.generalregression import GeneralRegressionNeuralNetwork
from gui.random_walk import RandomWalk
from gui.sarima import SARIMA
# from gui.elm import ELM
from gui.feature_selection import FS

# from gui.montecarlo import MonteCarlo
from gui.movingaverage import MovingAverage
# from gui.hybrid import Hybrid

import tkinter as tk
from tkinter import ttk

import warnings

warnings.filterwarnings("ignore")


class GUI:
    def __init__(self):
        self.gui = tk.Tk()
        self.parent = ttk.Notebook(self.gui)

        # test = Test()
        # self.add(test, "Test")

        time_series = TimeSeries()
        self.add(time_series, "Time Series")

        mlp = MultiLayerPerceptron()
        self.add(mlp, "MLP")

        svm = SupportVectorMachine()
        self.add(svm, "SVM")

        rf = RandomForest()
        self.add(rf, "Random Forest")

        lgbm = LGBM()
        self.add(lgbm, "LightGBM")

        catboost = CatBoost()
        self.add(catboost, "CatBoost")

        xgb = XGB()
        self.add(xgb, "XGBoost")

        ridge = Ridge()
        self.add(ridge, "Ridge Regression")

        lr = LinearModel()
        self.add(lr, "Linear Regression")

        grnn = GeneralRegressionNeuralNetwork()
        self.add(grnn, "GRNN")

        sarima = SARIMA()
        self.add(sarima, "SARIMA")

        # elm = ELM()
        # self.add(elm, "ELM")

        rw = RandomWalk()
        self.add(rw, "Random Walk")

        fs = FS()
        self.add(fs, "Feature Selection")

        # monte = MonteCarlo()
        # self.add(monte, "Monte Carlo")
        #
        moving_average = MovingAverage()
        self.add(moving_average, "Moving Average")
        #
        # hybrid = Hybrid()
        # self.add(hybrid, "Hybrid")

        self.parent.pack(expand=1, fill="both")

    def add(self, frame, text):
        self.parent.add(frame.root, text=text)

    def start(self):
        self.gui.mainloop()


s = GUI()
s.start()
