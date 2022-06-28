#!/usr/bin/env python

import argparse
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def load_data(
    file_path: Path | str, joint: int, n_delay: int
) -> tuple[np.ndarray, np.ndarray]:
    data_raw: pd.DataFrame
    assert n_delay > 0
    data_raw = pd.read_csv(file_path)  # type: ignore
    data = data_raw[:-n_delay]
    data_delayed = data_raw[n_delay:].reset_index(drop=True)
    q_dq_delayed = data_delayed[[f"q{joint}", f"dq{joint}"]]
    states = data[[f"q{joint}", f"dq{joint}", f"pa{joint}", f"pb{joint}"]]
    df_X = pd.concat([q_dq_delayed, states], axis=1)
    df_y = data_delayed[[f"ca{joint}", f"cb{joint}"]]
    return df_X.to_numpy(), df_y.to_numpy()


def load_data_from_directory(
    directory_path: Path | str, joint: int, n_delay: int
) -> tuple[np.ndarray, np.ndarray]:
    print("loading...")
    directory_path = Path(directory_path)
    if not directory_path.is_dir():
        RuntimeError(f"{str(directory_path)} must be directory")
    data_files = directory_path.glob("*.csv")
    X, y = np.empty(shape=(0, 6), dtype=float), np.empty(shape=(0, 2), dtype=float)
    notfound = True
    for data_file in data_files:
        _X, _y = load_data(data_file, joint, n_delay)
        X = np.vstack((X, _X))
        y = np.vstack((y, _y))
        notfound = False
    if notfound:
        warnings.warn(f"No data files found in {str(directory_path)}")
    if len(y.shape) == 2 and y.shape[1] == 1:
        y = y.ravel()
    return X, y


def learn(
    X_train: np.ndarray, y_train: np.ndarray, params: dict[str, Any]
) -> LinearRegression:
    print("training...")
    reg = LinearRegression(**params)
    reg.fit(X_train, y_train)
    return reg


def plot(
    reg: LinearRegression,
    X_test: np.ndarray,
    y_test: np.ndarray,
    index: int | Iterable[int] = 0,
    title: str | None = None,
):
    print("plotting...")
    y_predict = reg.predict(X_test)
    _, ax = plt.subplots()
    if not isinstance(index, Iterable):
        index = [index]
    for i in index:
        ax.plot(y_test[:, i], label="test")
        ax.plot(y_predict[:, i], label="predict")
    if title is not None:
        ax.set_title(title)
    ax.legend()


def parse():
    parser = argparse.ArgumentParser(
        description="Predict output using LinearRegression"
    )
    parser.add_argument(
        "--train-data",
        required=True,
        help="Path to directory where train data files are stored.",
    )
    parser.add_argument(
        "--test-data",
        required=True,
        help="Path to directory where test data files are stored.",
    )
    parser.add_argument("--joint", required=True, type=int, help="Joint index to move")
    parser.add_argument(
        "--n-delay", required=True, type=int, help="Num of samples delayed"
    )
    parser.add_argument(
        "--fit-intercept",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).",
    )
    parser.add_argument(
        "--copy-X",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If True, X will be copied; else, it may be overwritten.",
    )
    parser.add_argument(
        "--n-jobs",
        default=None,
        type=int,
        help="The number of jobs to use for the computation. This will only provide speedup in case of sufficiently large problems, that is if firstly n_targets > 1 and secondly X is sparse or if positive is set to True. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details.",
    )
    parser.add_argument(
        "--positive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When set to True, forces the coefficients to be positive. This option is only supported for dense arrays.",
    )
    return parser.parse_args()


def convert_args_to_reg_params(args):
    params = vars(args).copy()
    for k in ["train_data", "test_data", "joint", "n_delay"]:
        params.pop(k)
    return params


def main():
    args = parse()
    params = convert_args_to_reg_params(args)
    X_train, y_train = load_data_from_directory(
        args.train_data, args.joint, args.n_delay
    )
    X_test, y_test = load_data_from_directory(args.test_data, args.joint, args.n_delay)
    reg = learn(X_train, y_train, params)
    plot(reg, X_test, y_test, [0, 1], f"pressure at valve (joint: {args.joint})")
    plt.show()


if __name__ == "__main__":
    main()
