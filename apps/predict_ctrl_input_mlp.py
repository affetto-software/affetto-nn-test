#!/usr/bin/env python

import argparse
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
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
    return fit_data(X_train, y_train, args.scaler, MLPRegressor, params)


def plot(
    args: argparse.Namespace,
    reg: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sfparam: dict[str, Any],
) -> None:
    print("plotting...")
    suffix = f" (joint={args.joint}, k={args.n_predict}, scaler={args.scaler})"
    plot_prediction(
        reg, X_test, y_test, index=[0, 1], title="pressure at valve" + suffix, **sfparam
    )
    print(f"score={reg.score(X_test, y_test)}")
    if not args.noshow:
        plt.show()


def parse():
    parser = argparse.ArgumentParser(
        description="Predict control inputs using MLPRegressor."
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
    keys = [
        "hidden_layer_sizes",
        "activation",
        "solver",
        "alpha",
        "learning_rate",
        "learning_rate_init",
        "max_iter",
        "momentum",
        "nesterovs_momentum",
    ]
    params = {k: args_dict[k] for k in keys}
    v = params["hidden_layer_sizes"]
    params["hidden_layer_sizes"] = tuple(v)
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
