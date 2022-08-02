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
from model import ESN, Tikhonov


class Loader(LoaderBase):
    @staticmethod
    def reshape(
        joints: int | list[int],
        data: pd.DataFrame,
        data_for_predict: pd.DataFrame,
        data_for_control: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        labels = Loader.labels
        desired = data_for_predict[labels("q", joints) + labels("dq", joints)]
        states = data[
            labels("q", joints)
            + labels("dq", joints)
            + labels("pa", joints)
            + labels("pb", joints)
        ]
        ctrl = data_for_control[labels("ca", joints) + labels("cb", joints)]
        df_X = pd.concat([desired, states], axis=1)
        df_y = ctrl
        return df_X.to_numpy(), df_y.to_numpy()


def fit(
    args: argparse.Namespace,
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict[str, Any],
) -> ESN:
    print("fitting...")
    esn = ESN(X_train.shape[1], y_train.shape[1], **params)
    esn.train(X_train, y_train, Tikhonov(params["N_x"], y_train.shape[1], 1e-4))
    return esn


def plot(
    args: argparse.Namespace,
    esn: ESN,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sfparam: dict[str, Any],
) -> None:
    print("plotting...")
    suffix = f" (joint={args.joint}, k={args.n_predict}, scaler={args.scaler})"
    plot_prediction(
        esn, X_test, y_test, index=[0, 1], title="pressure at valve" + suffix, **sfparam
    )
    # print(f"score={reg.score(X_test, y_test)}")
    if not args.noshow:
        plt.show()


def parse():
    parser = argparse.ArgumentParser(
        description="Predict control inputs using Echo State Network."
    )
    parser.add_argument(
        "--train-data",
        nargs="+",
        help="Path to directory where train data files are stored.",
    )
    parser.add_argument(
        "--test-data",
        nargs="+",
        help="Path to directory where test data files are stored.",
    )
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
        "-j",
        "--joint",
        required=True,
        nargs="+",
        type=int,
        help="Joint index (list) to move.",
    )
    parser.add_argument(
        "--n-neurons",
        default=300,
        type=int,
        help="The number of neurons in the reservoir.",
    )
    parser.add_argument(
        "--density",
        default=0.05,
        type=float,
        help="Connectivity probability in the reservoir.",
    )
    parser.add_argument(
        "--input-scale",
        default=1.0,
        type=float,
        help="The input scale.",
    )
    parser.add_argument(
        "--rho",
        default=0.95,
        type=float,
        help="Spectral radius of the inter connection weight matrix.",
    )
    parser.add_argument(
        "--leaking-rate",
        default="1.0",
        type=float,
        help="Leaking rate of the neurons.",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        help="The noise level.",
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


def convert_args_to_esn_params(args):
    args_dict = vars(args).copy()
    keys = [
        "n_neurons",
        "density",
        "input_scale",
        "rho",
        "leaking_rate",
        "noise_level",
    ]
    params = {k: args_dict[k] for k in keys}
    params["N_x"] = params.pop("n_neurons")
    return params


def main():
    args = parse()
    params = convert_args_to_esn_params(args)
    sfparam = convert_args_to_sfparam(args)
    loader = Loader(args.joint, args.n_predict, args.n_ctrl_period)
    X_train, y_train = loader.load(args.train_data)
    X_test, y_test = loader.load(args.test_data)
    esn = fit(args, X_train, y_train, params)
    plot(args, esn, X_test, y_test, sfparam)


if __name__ == "__main__":
    main()
