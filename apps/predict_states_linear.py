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
from _plot import convert_args_to_sfparam, plot_prediction


class Loader(LoaderBase):
    @staticmethod
    def reshape(
        j: int,
        data: pd.DataFrame,
        data_for_predict: pd.DataFrame,
        data_for_control: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        states = data[[f"q{j}", f"dq{j}", f"pa{j}", f"pb{j}"]]
        ctrl = data_for_control[[f"ca{j}", f"ca{j}"]]
        df_X = pd.concat([states, ctrl], axis=1)
        df_y = data_for_predict[[f"q{j}", f"dq{j}", f"pa{j}", f"pb{j}"]]
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
    sfparam: dict[str, Any],
) -> None:
    print("plotting...")
    suffix = f" (joint={args.joint}, k={args.n_predict}, scaler={args.scaler})"
    plot_prediction(reg, X_test, y_test, index=0, title="position" + suffix, **sfparam)
    plot_prediction(reg, X_test, y_test, index=1, title="velocity" + suffix, **sfparam)
    plot_prediction(
        reg,
        X_test,
        y_test,
        index=[2, 3],
        title="pressure at actuator" + suffix,
        **sfparam,
    )
    print(f"score={reg.score(X_test, y_test)}")
    if not args.noshow:
        plt.show()


def parse():
    parser = argparse.ArgumentParser(
        description="Predict states using LinearRegression."
    )
    parser.add_argument(
        "--train-data",
        required=True,
        nargs="+",
        help="Path to directory where train data files are stored.",
    )
    parser.add_argument(
        "--test-data",
        required=True,
        nargs="+",
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
    parser.add_argument(
        "-d", "--basedir", default="fig", help="directory where figures will be saved"
    )
    parser.add_argument(
        "-e",
        "--extension",
        default=["png"],
        nargs="+",
        help="extensions to save as figures",
    )
    parser.add_argument(
        "-T", "--time", nargs="+", type=float, help="time range to show in figure"
    )
    parser.add_argument(
        "-s", "--savefig", action="store_true", help="export figures if specified"
    )
    parser.add_argument(
        "-x", "--noshow", action="store_true", help="do not show figures if specified"
    )
    return parser.parse_args()


def convert_args_to_reg_params(args):
    args_dict = vars(args).copy()
    keys = ["fit_intercept", "copy_X", "n_jobs"]
    params = {k: args_dict[k] for k in keys}
    return params


def main():
    args = parse()
    params = convert_args_to_reg_params(args)
    sfparam = convert_args_to_sfparam(args)
    loader = Loader(args.joint, args.n_predict, args.n_ctrl_period)
    X_train, y_train = loader.load(args.train_data)
    X_test, y_test = loader.load(args.test_data)
    reg = fit(args, X_train, y_train, params)
    plot(args, reg, X_test, y_test, sfparam)


if __name__ == "__main__":
    main()
