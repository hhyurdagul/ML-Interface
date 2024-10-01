import os
import sys
os.environ["TCL_LIBRARY"] = sys.base_prefix + "/lib/tcl8.6"
os.environ["TK_LIBRARY"] = sys.base_prefix + "/lib/tk8.6"

# from gui.timeseries import TimeSeries

# from gui.mlp import MultiLayerPerceptron
# from gui.supportvectormachine import SupportVectorMachine
from mlinterface.gui.random_forest import RandomForest
# from gui.lgbm import LGBM
# from gui.catboost_arch import CatBoost
# from gui.ridge import Ridge
# from gui.linear_regression import LinearModel
# from gui.generalregression import GeneralRegressionNeuralNetwork
# from gui.random_walk import RandomWalk
# from gui.sarima import SARIMA
# from gui.elm import ELM

# from gui.feature_selection import FS
# from gui.montecarlo import MonteCarlo
# from gui.movingaverage import MovingAverage
# from gui.hybrid import Hybrid

import tkinter as tk
from tkinter import ttk

import warnings

warnings.filterwarnings("ignore")


class GUI:
    def __init__(self):
        self.root = tk.Tk()
        parent = ttk.Notebook(self.root, padding="0i")
        parent.pack(expand=1, fill="both", padx=1, pady=1)

        # Time Series
        time_series = ttk.Frame(parent)
        parent.add(time_series, text="Time Series")

        time_series_models = ttk.Notebook(time_series, padding="0i")
        time_series_models.pack(expand=1, fill="both")

        rf = RandomForest()
        time_series_models.add(rf.root, text="Random Forest")

        # Regression
        regression = ttk.Frame(parent)
        parent.add(regression, text="Regression")

        regression_models = ttk.Notebook(regression, padding="0i")
        regression_models.pack(expand=1, fill="both")

        regression_models.add(ttk.Frame(), text="Empty")

        # Classification
        classification = ttk.Frame(parent)
        parent.add(classification, text="Classification")

        classification_models = ttk.Notebook(classification, padding="0i")
        classification_models.pack(expand=1, fill="both")

        classification_models.add(ttk.Frame(), text="Empty")

        # time_series = TimeSeries()
        # self.add(time_series, "Time Series")

        # mlp = MultiLayerPerceptron()
        # self.add(mlp, "MLP")

        # svm = SupportVectorMachine()
        # self.add(svm, "SVM")


        # lgbm = LGBM()
        # self.add(lgbm, "LightGBM")

        # catboost = CatBoost()
        # self.add(catboost, "CatBoost")

        # ridge = Ridge()
        # self.add(ridge, "Ridge Regression")

        # lr = LinearModel()
        # self.add(lr, "Linear Regression")

        # grnn = GeneralRegressionNeuralNetwork()
        # self.add(grnn, "GRNN")

        # sarima = SARIMA()
        # self.add(sarima, "SARIMA")

        # elm = ELM()
        # self.add(elm, "ELM")
        #
        # rw = RandomWalk()
        # self.add(rw, "Random Walk")
        #
        # fs = FS()
        # self.add(fs, "Feature Selection")

        # monte = MonteCarlo()
        # self.add(monte, "Monte Carlo")
        #
        # moving_average = MovingAverage()
        # self.add(moving_average, "Moving Average")
        #
        # hybrid = Hybrid()
        # self.add(hybrid, "Hybrid")




if __name__ == "__main__":
    GUI().root.mainloop()
