import numpy as np
from utils import shift_array


class Lookback:
    def __init__(
        self,
        lookback_count: int,
        seasonal_lookback_count: int,
        seasonal_lookback_period: int,
    ):
        match (seasonal_lookback_count, seasonal_lookback_period):
            case x, y if x > 0 and y <= 0:
                raise ValueError(
                    "Seasonal lookback count and period both have to be zero or greater than zero"
                )
            case x, y if x <= 0 and y > 0:
                raise ValueError(
                    "Seasonal lookback count and period both have to be zero or greater than zero"
                )

        self.lookback_count = lookback_count
        self.seasonal_lookback_count = seasonal_lookback_count
        self.seasonal_lookback_period = seasonal_lookback_period

        self.lookback_index = list(range(1, lookback_count + 1, 1))
        self.seasonal_lookback_index = list(
            range(
                seasonal_lookback_period,
                (seasonal_lookback_count + 1) * seasonal_lookback_period,
                max(seasonal_lookback_period, 1),
            )
        )

        self.merged_index = list(
            set(self.lookback_index + self.seasonal_lookback_index)
        )

    def __get_lookback_stack(self, y: np.ndarray, n):
        stack = y.copy().reshape(-1, 1)
        for i in range(1, n + 1):
            stack = np.hstack((stack, shift_array(y, i).reshape(-1, 1)))
        return stack

    def get_lookback(self, X: np.ndarray, y: np.ndarray):
        if self.merged_index:
            stack = self.__get_lookback_stack(y, max(self.merged_index))
            stack = stack[:, self.merged_index]
            X = np.concatenate((X, stack), axis=1)

            X = X[~np.isnan(X).any(axis=1)]
            y = y[y.shape[0] - X.shape[0] :]

        return X, y


X = np.random.rand(15, 4).round(2)
y = np.arange(15)


lookback = Lookback(9, 2, 2)
X, y = lookback.get_lookback(X, y)
print(X)
print(y)
