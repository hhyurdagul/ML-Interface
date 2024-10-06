from PySide6.QtWidgets import QGridLayout, QWidget, QVBoxLayout, QPushButton, QMessageBox
from PySide6.QtCore import Signal
from mlinterface.components.input_component import InputComponent
from mlinterface.components.time_series import PreprocessingComponent
from mlinterface.components.model_components.temp import ModelComponent

class RandomForest(QWidget):
    def __init__(self):
        super().__init__()
        # Create main layout
        layout = QGridLayout(self)

        # Add components to the layout
        input_component = InputComponent("Train Data Input", self.read_data)
        preprocessing_component = PreprocessingComponent("Preprocessing")
        random_forest_component = ModelComponent()


        layout.addWidget(input_component.root, 0, 0, 1, 1)
        layout.addWidget(preprocessing_component.root, 1, 0, 1, 1)
        layout.addWidget(random_forest_component.root, 0, 1, 1, 1)

        # Add stretch to push widgets to the top and distribute space
        self.setLayout(layout)

    # def setup_ui(self):
    #     # Add a "Process Data" button
    #     self.process_button = QPushButton("Process Data")
    #     self.process_button.clicked.connect(self.process_data)
    #     layout.addWidget(self.process_button)

    #     self.setLayout(layout)
    def read_data(self, file_path: str) -> list[str]:
        # Implement your data reading logic here
        # This should return a list of column names from your data file
        # For example:
        import pandas as pd
        df = pd.read_excel(file_path)
        return df.columns.tolist()
        pass
