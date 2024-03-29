import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Sequence

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

    @classmethod
    def labels(cls, label: str, joints: int | list[int]) -> list[str]:
        if not isinstance(joints, Iterable):
            joints = [joints]
        labels = [label + str(i) for i in joints]
        return labels

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

    def load(
        self, directory_path: Path | str | Sequence[Path] | Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        print("loading...")

        if isinstance(directory_path, (str | Path)):
            directory_path_list = [directory_path]
        elif isinstance(directory_path, Sequence):
            directory_path_list = directory_path
        else:
            raise RuntimeError(f"{str(directory_path)} cannot be handled")

        data_files = []
        for path in directory_path_list:
            path = Path(path)
            if path.is_dir():
                data_files = path.glob("*.csv")
            elif path.is_file():
                data_files = [path]
            else:
                raise RuntimeError(f"{str(path)} must be directory or file")

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
