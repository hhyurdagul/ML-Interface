import pandas as pd
from typing import Tuple, TypeAlias
from sklearn.preprocessing import StandardScaler, MinMaxScaler

ScalerType: TypeAlias = StandardScaler | MinMaxScaler

def scale_data(X: pd.DataFrame, y: pd.Series, scaler_choice: str) -> Tuple[
    pd.DataFrame,
    pd.Series,
    ScalerType,
    ScalerType
]:
    if scaler_choice == "StandardScaler":
        feature_scaler = StandardScaler()
        label_scaler = StandardScaler()
    elif scaler_choice == "MinMaxScaler":
        feature_scaler = MinMaxScaler()
        label_scaler = MinMaxScaler()

    X.iloc[:] = feature_scaler.fit_transform(X)
    y.iloc[:] = label_scaler.fit_transform(y.to_numpy().reshape(-1, 1))

    return X, y, feature_scaler, label_scaler
