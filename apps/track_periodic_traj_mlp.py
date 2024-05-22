#!/usr/bin/env python

import argparse
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _fit import fit_data
from _loader import LoaderBase
from _plot import convert_args_to_sfparam, plot_prediction
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

from affctrllib import AffComm, AffPosCtrl, AffStateThread, Logger, Timer

DEFAULT_CONFIG_PATH = Path(__file__).parent.joinpath("config.toml")
DEFAULT_JOINT_LIST = [0]
DEFAULT_DURATION = 20.0  # sec
DEFAULT_AMPLITUDE = 40.0
DEFAULT_PERIOD = 5.0
DEFAULT_BIAS = 50.0


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


def prepare_ctrl(
    config: str, sfreq: float | None, cfreq: float | None
) -> tuple[AffComm, AffPosCtrl, AffStateThread]:
    comm = AffComm(config_path=config)
    comm.create_command_socket()
    state = AffStateThread(
        config=config, freq=sfreq, logging=False, output=None, butterworth=False
    )
    ctrl = AffPosCtrl(config_path=config, freq=cfreq)
    state.prepare()
    state.start()
    print("Waiting until robot gets stationary...")
    time.sleep(3)
    return comm, ctrl, state


def prepare_logger(dof: int, output: str | Path) -> Logger:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger = Logger(output_path)
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


def create_const_trajectory(
    A: float, T: float, bias: float, joint: int, q0: np.ndarray
) -> tuple[Callable[[float], np.ndarray], Callable[[float], np.ndarray]]:
    _ = A
    _ = T

    def qdes_func(_: float) -> np.ndarray:
        q = np.copy(q0)
        q[0] = 50  # make waist joint keep at middle.
        q[joint] = bias
        return q

    def dqdes_func(_: float) -> np.ndarray:
        dq = np.zeros((len(q0),))
        return dq

    return qdes_func, dqdes_func


def create_sin_trajectory(
    A: float, T: float, b: float, joint: int, q0: np.ndarray
) -> tuple[Callable[[float], np.ndarray], Callable[[float], np.ndarray]]:
    def qdes_func(t: float) -> np.ndarray:
        q = np.copy(q0)
        q[0] = 50  # make waist joint keep at middle.
        q[joint] = A * np.sin(2.0 * np.pi * t / T) + b
        return q

    def dqdes_func(t: float) -> np.ndarray:
        dq = np.zeros((len(q0),))
        dq[joint] = 2.0 * np.pi * A * np.cos(2.0 * np.pi * t / T) / T
        return dq

    return qdes_func, dqdes_func


def create_step_trajectory(
    A: float, T: float, b: float, joint: int, q0: np.ndarray
) -> tuple[Callable[[float], np.ndarray], Callable[[float], np.ndarray]]:
    def qdes_func(t: float) -> np.ndarray:
        q = np.copy(q0)
        q[0] = 50  # make waist joint keep at middle.
        if t - int(t / T) * T < 0.5 * T:
            q[joint] = b + A
        else:
            q[joint] = b - A
        return q

    def dqdes_func(_: float) -> np.ndarray:
        dq = np.zeros((len(q0),))
        return dq

    return qdes_func, dqdes_func


def control(
    comm: AffComm,
    ctrl: AffPosCtrl,
    state: AffStateThread,
    qdes_func: Callable[[float], np.ndarray],
    dqdes_func: Callable[[float], np.ndarray],
    duration: float,
    logger: Logger | None = None,
    log_filename: str | Path | None = None,
):
    timer = Timer(rate=ctrl.freq)
    if logger is not None:
        logger.erase_data()
        if log_filename is not None:
            logger.fpath = log_filename
        elif logger.fpath is None:
            logger.fpath = Path("data.csv")
    timer.start()
    t = 0
    while t < duration:
        t = timer.elapsed_time()
        rq, rdq, rpa, rpb = state.get_raw_states()
        q, dq, pa, pb = state.get_states()
        qdes = qdes_func(t)
        dqdes = dqdes_func(t)
        ca, cb = ctrl.update(t, q, dq, pa, pb, qdes, dqdes)
        comm.send_commands(ca, cb)
        if logger is not None:
            logger.store(t, rq, rdq, rpa, rpb, q, dq, pa, pb, ca, cb, qdes, dqdes)
        print(f"\rt = {t:.2f} / {duration:.2f}", end="")
        timer.block()
    print()
    if logger is not None:
        logger.dump()


def control_reg(
    reg: Pipeline,
    comm: AffComm,
    ctrl: AffPosCtrl,
    state: AffStateThread,
    joint: int | list[int],
    qdes_func: Callable[[float], np.ndarray],
    dqdes_func: Callable[[float], np.ndarray],
    duration: float,
    time_offset: float,
    logger: Logger | None = None,
    log_filename: str | Path | None = None,
):
    timer = Timer(rate=ctrl.freq)
    if logger is not None:
        logger.erase_data()
        if log_filename is not None:
            logger.fpath = log_filename
        elif logger.fpath is None:
            logger.fpath = Path("data.csv")
    timer.start()
    t = 0
    if isinstance(joint, Iterable):
        j = joint[0]
    else:
        j = joint
    while t < duration:
        t = timer.elapsed_time()
        rq, rdq, rpa, rpb = state.get_raw_states()
        q, dq, pa, pb = state.get_states()
        qdes, qdes_offset = qdes_func(t), qdes_func(t + time_offset)
        dqdes, dqdes_offset = dqdes_func(t), dqdes_func(t + time_offset)
        ca, cb = ctrl.update(t, q, dq, pa, pb, qdes, dqdes)
        X = np.array([[qdes_offset[j], dqdes_offset[j], q[j], dq[j], pa[j], pb[j]]])
        y = np.ravel(reg.predict(X))
        ca[j], cb[j] = y[0], y[1]
        comm.send_commands(ca, cb)
        if logger is not None:
            logger.store(t, rq, rdq, rpa, rpb, q, dq, pa, pb, ca, cb, qdes, dqdes)
        print(f"\rt = {t:.2f} / {duration:.2f}", end="")
        timer.block()
    print()
    if logger is not None:
        logger.dump()


TRAJECTORY: dict[
    str,
    Callable[
        [float, float, float, int, np.ndarray],
        tuple[Callable[[float], np.ndarray], Callable[[float], np.ndarray]],
    ],
] = {
    "const": create_const_trajectory,
    "sin": create_sin_trajectory,
    "step": create_step_trajectory,
}


def mainloop(args: argparse.Namespace, reg: Pipeline | None = None):
    # Prepare controller.
    comm, ctrl, state = prepare_ctrl(args.config, args.sfreq, args.cfreq)

    # Prepare logger.
    logger = prepare_logger(ctrl.dof, args.output)

    # Get initial pose.
    q0 = state.q

    # Parameters for periodic trajectory.
    A = args.amplitude
    T = args.period
    b = args.bias

    try:
        # Get back to home position.
        print("Getting back to home position...")
        qdes_func, dqdes_func = TRAJECTORY["const"](A, T, b, args.joint, q0)
        control(comm, ctrl, state, qdes_func, dqdes_func, 3)

        # Track a periodic trajectory.
        time_offset = args.n_predict * ctrl.dt
        qdes_func, dqdes_func = TRAJECTORY[args.traj_type](A, T, b, args.joint, q0)
        if reg is None:
            control(
                comm,
                ctrl,
                state,
                qdes_func,
                dqdes_func,
                args.duration,
                logger=logger,
            )
        else:
            control_reg(
                reg,
                comm,
                ctrl,
                state,
                args.joint,
                qdes_func,
                dqdes_func,
                args.duration,
                time_offset,
                logger=logger,
            )
    finally:
        print("Quitting...")
        comm.close_command_socket()
        time.sleep(0.1)
        state.join()


def parse():
    parser = argparse.ArgumentParser(
        description="Make robot track periodic trajectory using MLPRegressor"
    )
    parser.add_argument(
        "--ctrl",
        choices=["pid", "mlp"],
        default="mlp",
        help="Controller type to be executed.",
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
        "--n-predict", required=True, type=int, help="Number of samples to predict"
    )
    parser.add_argument(
        "--n-ctrl-period",
        required=True,
        type=int,
        help="Number of samples that equals control period",
    )
    parser.add_argument(
        "--scaler",
        choices=["minmax", "maxabs", "robust", "std", "none"],
        default="minmax",
        help="The scaler for preprocessing",
    )
    parser.add_argument(
        "-c", "--config", default=DEFAULT_CONFIG_PATH, help="config file"
    )
    parser.add_argument("-o", "--output", help="output filename")
    parser.add_argument(
        "-j",
        "--joint",
        required=True,
        nargs="+",
        default=DEFAULT_JOINT_LIST,
        type=int,
        help="Joint index (list) to move",
    )
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
        "-t",
        "--trajectory",
        dest="traj_type",
        default="sin",
        choices=["sin", "step"],
        help="Trajectory type to be generated.",
    )
    parser.add_argument(
        "-D",
        "--time-duration",
        dest="duration",
        default=DEFAULT_DURATION,
        type=float,
        help="Time duration to execute.",
    )
    parser.add_argument(
        "-a",
        "--amplitude",
        default=DEFAULT_AMPLITUDE,
        type=float,
        help="Amplitude list for reference trajectory.",
    )
    parser.add_argument(
        "-p",
        "--period",
        default=DEFAULT_PERIOD,
        type=float,
        help="Period list for reference trajectory.",
    )
    parser.add_argument(
        "-b",
        "--bias",
        default=DEFAULT_BIAS,
        type=float,
        help="Bias list for reference trajectory.",
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
    parser.add_argument("-m", "--model", help="path to saved model")
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
    if args.ctrl == "mlp":
        loader = Loader(args.joint, args.n_predict, args.n_ctrl_period)
        X_train, y_train = loader.load(args.train_data)
        if args.model is None:
            reg = fit(args, X_train, y_train, params)
            joblib.dump(reg, "model.joblib")
        else:
            reg = joblib.load(args.model)
        if args.test_data is not None:
            X_test, y_test = loader.load(args.test_data)
            plot(args, reg, X_test, y_test, sfparam)
        else:
            mainloop(args, reg)
    else:
        mainloop(args, None)


if __name__ == "__main__":
    main()
