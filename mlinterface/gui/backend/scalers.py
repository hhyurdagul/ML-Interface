import numpy as np


# NOTE: Fuck inheritance

class ScalerParams:
    def __init__(self, data: np.ndarray) -> None:
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)

    def to_dict(self) -> dict[str, list[float]]:
        data = {
            "min": self.min.tolist(),
            "max": self.max.tolist(),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }
        return data

    def from_dict(self, params: dict[str, list[float]]) -> None:
        self.min = np.array(params["min"])
        self.max = np.array(params["max"])
        self.mean = np.array(params["mean"])
        self.std = np.array(params["std"])


class ScalerBase:
    def fit(self, data: np.ndarray) -> None:
        self.params = ScalerParams(data)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return data

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data

    def get_params(self) -> dict[str, list[float]]:
        return self.params.to_dict()

    def set_params(self, params: dict[str, list[float]]) -> None:
        self.params.from_dict(params)


class StandardScaler:
    def fit(self, data: np.ndarray) -> None:
        self.params = ScalerParams(data)

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.params.mean is None or self.params.std is None:
            raise ValueError("Fit the scaler before transforming data")
        return (data - self.params.mean) / self.params.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.params.mean is None or self.params.std is None:
            raise ValueError("Fit the scaler before inverse transforming data")
        return data * self.params.std + self.params.mean
    
    def get_params(self) -> dict[str, list[float]]:
        return self.params.to_dict()

    def set_params(self, params: dict[str, list[float]]) -> None:
        self.params.from_dict(params)



class MinMaxScaler(ScalerBase):
    def fit(self, data: np.ndarray) -> None:
        self.params = ScalerParams(data)

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.params.min is None or self.params.max is None:
            raise ValueError("Fit the scaler before transforming data")
        return (data - self.params.min) / (self.params.max - self.params.min)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.params.min is None or self.params.max is None:
            raise ValueError("Fit the scaler before inverse transforming data")
        return data * (self.params.max - self.params.min) + self.params.min
    
    def get_params(self) -> dict[str, list[float]]:
        return self.params.to_dict()

    def set_params(self, params: dict[str, list[float]]) -> None:
        self.params.from_dict(params)
