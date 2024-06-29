from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class ScalerParams:
    min: np.ndarray
    max: np.ndarray
    scale: np.ndarray
    min_scaled: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    def to_dict(self) -> dict[str, list[float]]:
        data: dict[str, np.ndarray] = asdict(self)
        return {i: j.tolist() for i, j in data.items()}


class ScalerBase:
    def fit(self, data: np.ndarray) -> "ScalerBase":
        self.params = ScalerParams(
            np.min(data, axis=0),
            np.max(data, axis=0),
            1 / (np.max(data, axis=0) - np.min(data, axis=0)),
            -np.min(data, axis=0) * 1 / (np.max(data, axis=0) - np.min(data, axis=0)),
            np.mean(data, axis=0),
            np.std(data, axis=0),
        )
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return data

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.fit(data).transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data

    def get_params(self) -> dict[str, list[float]]:
        return self.params.to_dict()

    def set_params(self, params: dict[str, list[float]]) -> "ScalerBase":
        data = {i: np.array(j) for i, j in params.items()}
        self.params = ScalerParams(**data)
        return self


class StandardScaler(ScalerBase):
    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.params.mean is None or self.params.std is None:
            raise ValueError("Fit the scaler before transforming data")
        return (data - self.params.mean) / self.params.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.params.mean is None or self.params.std is None:
            raise ValueError("Fit the scaler before inverse transforming data")
        return data * self.params.std + self.params.mean


class MinMaxScaler(ScalerBase):
    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.params.min is None or self.params.max is None:
            raise ValueError("Fit the scaler before transforming data")
        return data * self.params.scale + self.params.min_scaled

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.params.min is None or self.params.max is None:
            raise ValueError("Fit the scaler before inverse transforming data")
        return (data - self.params.min_scaled) / self.params.scale
