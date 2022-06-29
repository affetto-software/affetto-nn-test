from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

SCALER_MAP = {
    "minmax": MinMaxScaler(),
    "maxabs": MaxAbsScaler(),
    "robust": RobustScaler(),
    "std": StandardScaler(),
    "none": None,
}


def fit_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scaler_name: str,
    regressor: Any,
    params: dict[str, Any],
) -> Pipeline:
    print("fitting...")
    scaler = SCALER_MAP[scaler_name]
    reg = make_pipeline(scaler, regressor(**params))
    reg.fit(X_train, y_train)
    return reg
