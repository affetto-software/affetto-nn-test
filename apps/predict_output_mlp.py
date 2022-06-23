#!/usr/bin/env python

import argparse
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor


def load_data(
    file_path: Path | str, joint: int, n_delay: int
) -> tuple[np.ndarray, np.ndarray]:
    data_raw: pd.DataFrame
    assert n_delay > 0
    data_raw = pd.read_csv(file_path)  # type: ignore
    data = data_raw[:-n_delay]
    data_delayed = data_raw[n_delay:]
    labels_X = [
        f"q{joint}",
        f"dq{joint}",
        f"pa{joint}",
        f"pb{joint}",
        f"ca{joint}",
        f"cb{joint}",
    ]
    labels_y = [f"q{joint}"]
    df_X = data[labels_X]
    df_y = data_delayed[labels_y]
    return df_X.to_numpy(), df_y.to_numpy()


def load_data_from_directory(
    directory_path: Path | str, joint: int, n_delay: int
) -> tuple[np.ndarray, np.ndarray]:
    print("loading...")
    directory_path = Path(directory_path)
    if not directory_path.is_dir():
        RuntimeError(f"{str(directory_path)} must be directory")
    data_files = directory_path.glob("*.csv")
    X, y = np.empty(shape=(0, 6), dtype=float), np.empty(shape=(0, 1), dtype=float)
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
) -> MLPRegressor:
    print("training...")
    mlp = MLPRegressor(**params)
    mlp.fit(X_train, y_train)
    return mlp


def plot(mlp: MLPRegressor, X_test: np.ndarray, y_test: np.ndarray):
    print("plotting...")
    y_predict = mlp.predict(X_test)
    _, ax = plt.subplots()
    ax.plot(y_test, label="test")
    ax.plot(y_predict, label="predict")
    ax.legend()
    plt.show()


def parse():
    parser = argparse.ArgumentParser(description="Predict output using MLPRegressor")
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
        "--hidden-layer-sizes",
        default=(100,),
        nargs="+",
        type=int,
        help="The ith element represents the number of neurons in the ith hidden layer.",
    )
    parser.add_argument(
        "--activation",
        choices=["identity", "logistic", "tanh", "relu"],
        default="relu",
        help="Activation function for the hidden layer.",
    )
    parser.add_argument(
        "--solver",
        choices=["lbfgs", "sgd", "adam"],
        default="adam",
        help="The solver for weight optimization.",
    )
    parser.add_argument(
        "--alpha",
        default=0.0001,
        type=float,
        help="Strength of the L2 regularization term.",
    )
    parser.add_argument(
        "--learning-rate",
        choices=["constant", "invscaling", "adaptive"],
        default="constant",
        help="Learning rate schedule for weight updates.",
    )
    parser.add_argument(
        "--learning-rate-init",
        default=0.001,
        type=float,
        help="The initial learning rate used.",
    )
    parser.add_argument(
        "--max-iter",
        default=200,
        type=int,
        help="Maximum number of iterations.",
    )
    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        help="Momentum for gradient descent update. Should be between 0 and 1.",
    )
    parser.add_argument(
        "--nesterovs-momentum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use Nesterov’s momentum.",
    )
    return parser.parse_args()


def convert_args_to_mlp_params(args):
    params = vars(args).copy()
    for k in ["train_data", "test_data", "joint", "n_delay"]:
        params.pop(k)
    v = params["hidden_layer_sizes"]
    params["hidden_layer_sizes"] = tuple(v)
    return params


def main():
    args = parse()
    params = convert_args_to_mlp_params(args)
    X_train, y_train = load_data_from_directory(
        args.train_data, args.joint, args.n_delay
    )
    X_test, y_test = load_data_from_directory(args.test_data, args.joint, args.n_delay)
    mlp = learn(X_train, y_train, params)
    plot(mlp, X_test, y_test)


if __name__ == "__main__":
    main()
