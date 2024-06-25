import numpy as np


class ScalerBase:
    def fit(self, data: np.ndarray) -> "ScalerBase":
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)
        self.scale = 1 / (self.max - self.min)
        self.min_scaled = -self.min * self.scale

        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return data

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.fit(data).transform(data)

    def inverse_transform(self, data):
        pass

    def get_params(self) -> dict[str, list[float]]:
        return {
            "min": self.min.tolist(),
            "max": self.max.tolist(),
            "scale": self.scale.tolist(),
            "min_scaled": self.min_scaled.tolist(),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    def set_params(self, params: dict[str, list[float]]) -> "ScalerBase":
        self.min = np.array(params["min"])
        self.max = np.array(params["max"])
        self.scale = np.array(params["scale"])
        self.min_scaled = np.array(params["min_scaled"])

        self.mean = np.array(params["mean"])
        self.std = np.array(params["std"])
        return self

class StandardScaler(ScalerBase):
    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Fit the scaler before transforming data")
        return (data - self.mean) / self.std
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Fit the scaler before inverse transforming data")
        return data * self.std + self.mean


class MinMaxScaler(ScalerBase):
    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.min is None or self.max is None:
            raise ValueError("Fit the scaler before transforming data")
        return data * self.scale + self.min_scaled

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.min is None or self.max is None:
            raise ValueError("Fit the scaler before inverse transforming data")
        return (data - self.min_scaled) / self.scale

