import pandas as pd
from os import listdir

for i in listdir():
    if i.endswith(".csv"):
        df = pd.read_csv(i, index_col="Date")
        df[:-100].to_csv(i[:-4] + "_train.csv")
        df[-100:].to_csv(i[:-4] + "_test.csv")
