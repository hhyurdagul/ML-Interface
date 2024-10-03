from dataclasses import dataclass
from typing import Any
import numpy as np

class ModelConfig:

    @classmethod
    def get_random_forest_model(cls) -> RandomForestRegressor:
        return RandomForestRegressor()



class ModelHandler:
    def __init__(self, model_type: str, model_params: dict[str, Any]) -> None:
        self.model_type = model_type
        self.model_params = model_params
