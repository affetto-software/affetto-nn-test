#!/usr/bin/env python

import argparse
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from _fit import fit_data
from _loader import LoaderBase
from _plot import plot_prediction


class Loader(LoaderBase):
    @staticmethod
    def reshape(
        j: int,
        data: pd.DataFrame,
        data_for_predict: pd.DataFrame,
        data_for_control: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        desired = data_for_predict[[f"q{j}", f"dq{j}"]]
        states = data[[f"q{j}", f"dq{j}", f"pa{j}", f"pb{j}"]]
        ctrl = data_for_control[[f"ca{j}", f"cb{j}"]]
        df_X = pd.concat([desired, states], axis=1)
        df_y = ctrl
        return df_X.to_numpy(), df_y.to_numpy()


def fit(
    args: argparse.Namespace,
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict[str, Any],
) -> Pipeline:
    return fit_data(X_train, y_train, args.scaler, LinearRegression, params)


def plot(
    args: argparse.Namespace,
    reg: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    print("plotting...")
    suffix = f" (joint: {args.joint}, k: {args.n_predict}, scaler: {args.scaler})"
    plot_prediction(reg, X_test, y_test, [0, 1], "pressure at valve" + suffix)
    print(f"score={reg.score(X_test, y_test)}")
    plt.show()


def parse():
    parser = argparse.ArgumentParser(
        description="Predict control inputs using LinearRegression."
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
    parser.add_argument("--joint", required=True, type=int, help="Joint index to move.")
    parser.add_argument(
        "--n-predict", required=True, type=int, help="Number of samples for prediction."
    )
    parser.add_argument(
        "--n-ctrl-period",
        default=0,
        type=int,
        help="Number of samples that equals control period.",
    )
    parser.add_argument(
        "--scaler",
        choices=["minmax", "maxabs", "robust", "std", "none"],
        default="minmax",
        help="The scaler for preprocessing",
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
    for k in [
        "train_data",
        "test_data",
        "joint",
        "n_predict",
        "n_ctrl_period",
        "scaler",
    ]:
        params.pop(k)
    return params


def main():
    args = parse()
    params = convert_args_to_reg_params(args)
    loader = Loader(args.joint, args.n_predict, args.n_ctrl_period)
    X_train, y_train = loader.load(args.train_data)
    X_test, y_test = loader.load(args.test_data)
    reg = fit(args, X_train, y_train, params)
    plot(args, reg, X_test, y_test)


if __name__ == "__main__":
    main()
