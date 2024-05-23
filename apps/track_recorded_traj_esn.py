#!/usr/bin/env python

import argparse
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _loader import LoaderBase
from _plot import convert_args_to_sfparam, plot_prediction
from model import ESN, Tikhonov
from track_recorded_traj_mlp import Spline

from affctrllib import AffComm, AffPosCtrl, AffStateThread, Logger, Timer

DEFAULT_CONFIG_PATH = Path(__file__).parent.joinpath("config.toml")
DEFAULT_JOINT_LIST = [0]
DEFAULT_DURATION = 20.0  # sec


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
    _: argparse.Namespace,
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
    qdes: float | None, joint: int | list[int] | None, q0: np.ndarray
) -> tuple[Callable[[float], np.ndarray], Callable[[float], np.ndarray]]:
    def qdes_func(_: float) -> np.ndarray:
        q = np.copy(q0)
        q[0] = 50  # make waist joint keep at middle.
        if qdes is not None and joint is not None:
            q[joint] = qdes
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


def control_esn(
    esn: ESN,
    comm: AffComm,
    ctrl: AffPosCtrl,
    state: AffStateThread,
    joints: list[int],
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
    while t < duration:
        t = timer.elapsed_time()
        rq, rdq, rpa, rpb = state.get_raw_states()
        q, dq, pa, pb = state.get_states()
        qdes, qdes_offset = qdes_func(t), qdes_func(t + time_offset)
        dqdes, dqdes_offset = dqdes_func(t), dqdes_func(t + time_offset)
        ca, cb = ctrl.update(t, q, dq, pa, pb, qdes, dqdes)
        X = np.atleast_2d(
            np.concatenate(
                (
                    qdes_offset[joints],
                    dqdes_offset[joints],
                    q[joints],
                    dq[joints],
                    pa[joints],
                    pb[joints],
                )
            )
        )
        y = np.ravel(esn.predict(X))
        n = len(joints)
        ca[joints] = y[:n]
        cb[joints] = y[n:]
        comm.send_commands(ca, cb)
        if logger is not None:
            logger.store(t, rq, rdq, rpa, rpb, q, dq, pa, pb, ca, cb, qdes, dqdes)
        print(f"\rt = {t:.2f} / {duration:.2f}", end="")
        timer.block()
    print()
    if logger is not None:
        logger.dump()


def mainloop(args: argparse.Namespace, esn: ESN | None = None):
    # Prepare controller.
    comm, ctrl, state = prepare_ctrl(args.config, args.sfreq, args.cfreq)

    # Prepare logger.
    logger = prepare_logger(ctrl.dof, args.output)

    # Get initial pose.
    q0 = state.q

    # Load recorded data.
    print("Loading recorded data...")
    spline = Spline(args.record_data, args.joint)

    try:
        # Get the recorded trajectory.
        qdes_func = spline.get_qdes_func(q0)
        dqdes_func = spline.get_dqdes_func(q0)
        q0 = qdes_func(0)

        # Get back to home position.
        print("Getting back to home position...")
        qdes_func, dqdes_func = create_const_trajectory(None, None, q0)
        control(comm, ctrl, state, qdes_func, dqdes_func, 3)

        # Track a periodic trajectory.
        time_offset = args.n_predict * ctrl.dt
        qdes_func = spline.get_qdes_func(q0)
        dqdes_func = spline.get_dqdes_func(q0)
        if esn is None:
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
            control_esn(
                esn,
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
        description="Make robot track recorded trajectory using Echo State Network"
    )
    parser.add_argument(
        "--record-data",
        required=True,
        help="Path to file that contains recorded trajectory.",
    )
    parser.add_argument(
        "--ctrl",
        choices=["pid", "esn"],
        default="esn",
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
        "--n-predict", default=10, type=int, help="Number of samples to predict"
    )
    parser.add_argument(
        "--n-ctrl-period",
        default=1,
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
    if args.ctrl == "esn":
        loader = Loader(args.joint, args.n_predict, args.n_ctrl_period)
        X_train, y_train = loader.load(args.train_data)
        esn = fit(args, X_train, y_train, params)
        if args.test_data is not None:
            X_test, y_test = loader.load(args.test_data)
            plot(args, esn, X_test, y_test, sfparam)
        else:
            mainloop(args, esn)
    else:
        mainloop(args, None)


if __name__ == "__main__":
    main()
