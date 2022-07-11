import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class LoaderBase(ABC):
    def __init__(
        self,
        joint: int | list[int],
        n_predict: int,
        n_ctrl_period: int = 0,
    ) -> None:
        assert n_predict > 0
        assert n_ctrl_period >= 0
        assert n_predict >= n_ctrl_period
        self._joint = joint
        self._n_predict = n_predict
        self._n_ctrl_period = n_ctrl_period

    @staticmethod
    @abstractmethod
    def reshape(
        joint: int | list[int],
        data: pd.DataFrame,
        data_for_predict: pd.DataFrame,
        data_for_control: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def load_file(self, filepath: Path) -> tuple[np.ndarray, np.ndarray]:
        raw_data: pd.DataFrame
        raw_data = pd.read_csv(filepath)  # type: ignore
        data = raw_data[: -self._n_predict]
        data_for_predict = raw_data[self._n_predict :].reset_index(drop=True)
        if self._n_ctrl_period == self._n_predict:
            data_for_control = raw_data[self._n_ctrl_period :]
        else:
            end = -(self._n_predict - self._n_ctrl_period)
            data_for_control = raw_data[self._n_ctrl_period : end]
        data_for_control = data_for_control.reset_index(drop=True)
        return self.reshape(self._joint, data, data_for_predict, data_for_control)

    def load(self, directory_path: Path | str) -> tuple[np.ndarray, np.ndarray]:
        print("loading...")
        directory_path = Path(directory_path)
        if not directory_path.is_dir():
            RuntimeError(f"{str(directory_path)} must be directory")
        data_files = directory_path.glob("*.csv")
        X, y = np.empty(shape=(0, 6), dtype=float), np.empty(shape=(0, 2), dtype=float)
        notfound = True
        for data_file in data_files:
            _X, _y = self.load_file(data_file)
            if notfound:
                X = np.copy(_X)
                y = np.copy(_y)
            else:
                X = np.vstack((X, _X))
                y = np.vstack((y, _y))
            notfound = False
        if notfound:
            warnings.warn(f"No data files found in {str(directory_path)}")
        if len(y.shape) == 2 and y.shape[1] == 1:
            y = y.ravel()
        return X, y
