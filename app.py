import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout
# from mlinterface.gui.random_forest_qt import RandomForest
from mlinterface.pages.time_series import RandomForest

import warnings
warnings.filterwarnings("ignore")

class GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ML Interface")
        self.setGeometry(100, 100, 1280, 720)

        # Create main tab widget
        self.main_tabs = QTabWidget()
        self.setCentralWidget(self.main_tabs)

        # Time Series tab
        time_series_tab = QWidget()
        time_series_layout = QVBoxLayout()
        time_series_tab.setLayout(time_series_layout)

        time_series_models = QTabWidget()
        time_series_layout.addWidget(time_series_models)

        rf = RandomForest()
        time_series_models.addTab(rf, "Random Forest")

        self.main_tabs.addTab(time_series_tab, "Time Series")

        # Regression tab
        regression_tab = QWidget()
        regression_layout = QVBoxLayout()
        regression_tab.setLayout(regression_layout)

        regression_models = QTabWidget()
        regression_layout.addWidget(regression_models)

        regression_models.addTab(QWidget(), "Empty")

        self.main_tabs.addTab(regression_tab, "Regression")

        # Classification tab
        classification_tab = QWidget()
        classification_layout = QVBoxLayout()
        classification_tab.setLayout(classification_layout)

        classification_models = QTabWidget()
        classification_layout.addWidget(classification_models)

        classification_models.addTab(QWidget(), "Empty")

        self.main_tabs.addTab(classification_tab, "Classification")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = GUI()
    gui.show()
    sys.exit(app.exec_())
