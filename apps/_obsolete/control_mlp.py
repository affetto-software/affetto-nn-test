#!/usr/bin/env python

import argparse
import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from affctrllib import AffComm, AffPosCtrl, AffStateThread, Logger, Timer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

DEFAULT_CONFIG_PATH = Path(__file__).parent.joinpath("config.toml")


def load_data(
    file_path: Path | str, joint: int, n_predict: int, n_ctrl_period: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    raw_data: pd.DataFrame
    assert n_ctrl_period > 0
    assert n_predict >= n_ctrl_period
    raw_data = pd.read_csv(file_path)  # type: ignore

    data = raw_data[:-n_predict]
    data_for_predict = raw_data[n_predict:].reset_index(drop=True)
    if n_ctrl_period == n_predict:
        data_for_control = raw_data[n_ctrl_period:]
    else:
        end = -(n_predict - n_ctrl_period)
        data_for_control = raw_data[n_ctrl_period:end]
    data_for_control = data_for_control.reset_index(drop=True)
    desired = data_for_predict[[f"q{joint}", f"dq{joint}"]]
    states = data[[f"q{joint}", f"dq{joint}", f"pa{joint}", f"pb{joint}"]]
    df_X = pd.concat([desired, states], axis=1)
    df_y = data_for_control[[f"ca{joint}", f"cb{joint}"]]
    return df_X.to_numpy(), df_y.to_numpy()


def load_data_from_directory(
    directory_path: Path | str, joint: int, n_predict: int, n_ctrl_period: int
) -> tuple[np.ndarray, np.ndarray]:
    directory_path = Path(directory_path)
    if not directory_path.is_dir():
        RuntimeError(f"{str(directory_path)} must be directory")
    data_files = directory_path.glob("*.csv")
    X, y = np.empty(shape=(0, 6), dtype=float), np.empty(shape=(0, 2), dtype=float)
    notfound = True
    for data_file in data_files:
        _X, _y = load_data(data_file, joint, n_predict, n_ctrl_period)
        X = np.vstack((X, _X))
        y = np.vstack((y, _y))
        notfound = False
    if notfound:
        warnings.warn(f"No data files found in {str(directory_path)}")
    if len(y.shape) == 2 and y.shape[1] == 1:
        y = y.ravel()
    return X, y


def train(
    args: argparse.Namespace, params: dict[str, Any]
) -> tuple[MLPRegressor, MinMaxScaler]:
    print("training...")
    scaler = MinMaxScaler()
    X_train, y_train = load_data_from_directory(
        args.train_data, args.joint, args.n_predict, args.n_ctrl_period
    )
    X_train = scaler.fit_transform(X_train)
    mlp = MLPRegressor(**params)
    mlp.fit(X_train, y_train)
    return mlp, scaler


def plot(mlp: MLPRegressor, scaler: MinMaxScaler, args: argparse.Namespace):
    print("plotting...")
    X_test, y_test = load_data_from_directory(
        args.test_data, args.joint, args.n_predict, args.n_ctrl_period
    )
    X_test = scaler.transform(X_test)
    y_predict = mlp.predict(X_test)
    _, ax = plt.subplots()
    ax.plot(y_test, label="test")
    ax.plot(y_predict, label="predict")
    ax.legend()
    plt.show()


def prepare_controller(
    config: str | Path, sfreq: float | None, cfreq: float | None
) -> tuple[AffComm, AffPosCtrl, AffStateThread]:
    comm = AffComm(config_path=config)
    comm.create_command_socket()
    ctrl = AffPosCtrl(config_path=config, freq=cfreq)
    state = AffStateThread(config=config, freq=sfreq, logging=False, output=None)
    state.prepare()
    state.start()
    print("Wait until the robot gets stationary.")
    time.sleep(3)
    return comm, ctrl, state


def prepare_logger(dof: int, output: str | Path | None = None) -> Logger:
    logger = Logger(output)
    logger.set_labels(
        "t",
        # raw data
        [f"rq{i}" for i in range(dof)],
        [f"rdq{i}" for i in range(dof)],
        [f"rpa{i}" for i in range(dof)],
        [f"rpb{i}" for i in range(dof)],
        # estimated states
        [f"q{i}" for i in range(dof)],
        [f"dq{i}" for i in range(dof)],
        [f"pa{i}" for i in range(dof)],
        [f"pb{i}" for i in range(dof)],
        # command data
        [f"ca{i}" for i in range(dof)],
        [f"cb{i}" for i in range(dof)],
        [f"qdes{i}" for i in range(dof)],
        [f"dqdes{i}" for i in range(dof)],
    )
    return logger


def sinusoidal(t: float, A: float, T: float, phi: float, b: float) -> float:
    return A * np.sin(2.0 * np.pi * t / T - phi) + b


def step(t: float, A: float, T: float, phi: float, b: float) -> float:
    if t - int(t / T) * T - phi < 0.5 * T:
        return b + A
    else:
        return b - A


def control(mlp: MLPRegressor, args: argparse.Namespace):
    comm, ctrl, state = prepare_controller(args.config, args.sfreq, args.cfreq)
    logger = prepare_logger(ctrl.dof, args.output)
    timer = Timer(rate=args.cfreq)
    j = args.joint
    time_offset = args.n_predict * ctrl.dt
    q0 = state.q
    zeros = np.zeros(shape=(len(q0),))

    A = 40
    T = 5
    b = 50

    t = 0
    timer.start()
    try:
        while t < 21:
            t = timer.elapsed_time()
            rq, rdq, rpa, rpb = state.get_raw_states()
            q, dq, pa, pb = state.get_states()
            ca, cb = ctrl.update(t, q, dq, pa, pb, q0, zeros)
            qdes = q0.copy()
            qdes[j] = sinusoidal(t + time_offset, A, T, 0, b)
            dqdes = zeros.copy()
            dqdes[j] = sinusoidal(
                t + time_offset, A * T / (2.0 * np.pi), T, -0.5 * np.pi, b
            )
            X = np.array([[qdes[j], dqdes[j], q[j], dq[j], pa[j], pb[j]]])
            y = np.ravel(mlp.predict(X))
            # ca[j], cb[j] = ctrl.scale(y[0] + 50, y[1] + 50)
            ca[j], cb[j] = y[0], y[1]
            comm.send_commands(ca, cb)
            logger.store(t, rq, rdq, rpa, rpb, q, dq, pa, pb, ca, cb, qdes, dqdes)
            timer.block()

    finally:
        print("Quitting...")
        comm.close_command_socket()
        logger.dump()
        time.sleep(0.1)
        state.join()


def parse():
    parser = argparse.ArgumentParser(description="Predict output using MLPRegressor")
    parser.add_argument(
        "--train-data",
        required=True,
        help="Path to directory where train data files are stored.",
    )
    parser.add_argument(
        "--test-data",
        help="Path to directory where test data files are stored.",
    )
    parser.add_argument("--joint", required=True, type=int, help="Joint index to move")
    parser.add_argument(
        "--n-predict", required=True, type=int, help="Number of samples to predict"
    )
    parser.add_argument(
        "--n-ctrl-period",
        required=True,
        type=int,
        help="Number of samples that equals control period",
    )
    parser.add_argument(
        "-c", "--config", default=DEFAULT_CONFIG_PATH, help="config file"
    )
    parser.add_argument("-o", "--output", help="output filename")
    parser.add_argument(
        "-F",
        "--sensor-freq",
        dest="sfreq",
        type=float,
        help="sensor frequency",
    )
    parser.add_argument(
        "-f",
        "--control-freq",
        dest="cfreq",
        type=float,
        help="control frequency",
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
    for k in [
        "train_data",
        "test_data",
        "joint",
        "n_predict",
        "n_ctrl_period",
        "config",
        "output",
        "sfreq",
        "cfreq",
    ]:
        params.pop(k)
    v = params["hidden_layer_sizes"]
    params["hidden_layer_sizes"] = tuple(v)
    return params


def main():
    args = parse()
    params = convert_args_to_mlp_params(args)
    mlp, scaler = train(args, params)
    if args.test_data is not None:
        plot(mlp, scaler, args)
    else:
        control(mlp, args)


if __name__ == "__main__":
    main()
