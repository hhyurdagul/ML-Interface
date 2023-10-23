import pandas as pd
from typing import List

def handle_errors(*functions):
    for func in functions:
        if not func():
            return False
    return True


class DataHandler:
    def __init__(self):
        self.df = pd.DataFrame()
        self.df_read = False

    def read_data(self, file_path: str) -> List[str]:
        if file_path.endswith(".csv"):
            self.df = pd.read_csv(file_path)
        else:
            try:
                self.df = pd.read_excel(file_path)
            except Exception:
                self.df = pd.read_excel(file_path, engine="openpyxl")

        self.df_read = True

        return self.df.columns.to_list()
