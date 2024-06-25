import numpy as np
from utils import shift_array, concat_X_y




class Lookback:
    def __init__(self):

        pass

    def get_lookback(self, X, y, lookback):
        pass






X = np.random.rand(10, 4).round(2)
y = np.arange(10)


for i in range(1, 4):
    X = np.concatenate((X, shift_array(y, i).reshape(-1, 1)), axis=1)

print(X)
print(y)

