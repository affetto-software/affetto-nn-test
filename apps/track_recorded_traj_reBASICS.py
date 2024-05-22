#!/usr/bin/env python

import argparse
import time
from pathlib import Path
from typing import Any, Callable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _loader import LoaderBase
from _plot import convert_args_to_sfparam, savefig
from model import RLS, reBASICS
from track_recorded_traj_mlp import Spline

from affctrllib import AffComm, AffPosCtrl, AffStateThread, Logger, Timer

DEFAULT_CONFIG_PATH = Path(__file__).parent.joinpath("config.toml")
DEFAULT_JOINT_LIST = [0]
DEFAULT_DURATION = 20.0  # sec


np.random.seed(seed=0)


def pulse(
    T: int | float,
    value: float,
    width: int | float,
    start: int | float = 0.0,
    base: float = 0.0,
    dt: float | None = None,
):
    if dt is not None:
        T = int(T / dt)
        width = int(width / dt)
        start = int(start / dt)
    else:
        T = int(T)
        width = int(width)
        start = int(start)
    x = np.ones((T,), dtype=float) * base
    x[start : start + width] = value
    return x.reshape(-1, 1)


def pulse_data(
    y: np.ndarray,
    value: float,
    width: float,
    warmup: float,
    base: float = 0.0,
    dt: float = 0.001,
):
    n_warmup = int(warmup / dt)
    n_train = len(y)
    n_width = int(width / dt)
    return pulse(
        n_warmup + n_train,
        value,
        n_width,
        n_warmup - n_width,
        base=base,
        dt=1,
    )


def pulse_func(value: float, width: float, start: float, base: float = 0.0):
    def _pulse_func(t: float) -> np.ndarray:
        x = np.ones((1,), dtype=float) * base
        if t >= start and t < start + width:
            x[0] = value
        return x

    return _pulse_func


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
) -> reBASICS:
    print("fitting...")
    dt = 1.0 / args.cfreq
    warmup = args.warmup
    pulse_value = args.pulse_value
    pulse_duration = args.pulse_duration
    X_train = pulse_data(y_train, pulse_value, pulse_duration, warmup, dt=dt)

    model = reBASICS(X_train.shape[1], y_train.shape[1], **params)
    opt = RLS(params["N_module"], y_train.shape[1], delta=1.0, lam=1.0, update=1)

    duration = args.duration
    model.resample_inactive_module(
        X_train,
        threshold=0.01,
        span=(int((warmup + 0.5 * duration) / dt), None),
    )

    n_train_loop = args.n_train_loop
    n_warmup = int(warmup / dt)
    zeros = np.zeros((n_warmup, y_train.shape[1]))
    y_train = np.vstack((zeros, y_train))
    assert len(X_train) == len(y_train)

    for i in range(n_train_loop):
        print(f"Training ({i + 1}/{n_train_loop})")
        model.reset_reservoir_state(randomize_initial_state=True)
        model.adapt(X_train, y_train, opt, warmup=int(warmup / dt))
    return model


def plot(
    args: argparse.Namespace,
    model: reBASICS,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sfparam: dict[str, Any],
) -> None:
    print("plotting...")
    dt = 1.0 / args.cfreq
    warmup = args.warmup
    pulse_value = args.pulse_value
    pulse_duration = args.pulse_duration
    X_test = pulse_data(y_test, pulse_value, pulse_duration, warmup, dt=dt)

    n_warmup = int(warmup / dt)
    y_predict = model.run(X_test, warmup=n_warmup)
    y_predict = y_predict[n_warmup:]
    assert len(y_test) == len(y_predict)

    for i, joint in enumerate(args.joint):
        fig, ax = plt.subplots()
        # suffix = f" (joint={args.joint}, k={args.n_predict}, scaler={args.scaler})"
        title = f"pressure at valve (density={args.density:.3f}, rho={args.rho:.2f}, lr={args.leaking_rate:.2f}, joint={joint:02})"
        for j in (0, 1):
            col = 2 * i + j
            (line,) = ax.plot(y_test[:, col], ls="--", label="expected")
            ax.plot(
                y_predict[:, col],  # type: ignore
                c=line.get_color(),
                ls="-",
                label="predict",
            )
        ax.set_title(title)
        ax.legend()
        sfparam["filename"] = (
            title.translate(
                str.maketrans({" ": "_", "=": "-", "(": None, ")": None, ",": None})
            )
            + ".svg"
        )
        savefig(fig, **sfparam)

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


def control_reBASICS(
    model: reBASICS,
    comm: AffComm,
    ctrl: AffPosCtrl,
    state: AffStateThread,
    joints: list[int],
    qdes_func: Callable[[float], np.ndarray],
    dqdes_func: Callable[[float], np.ndarray],
    duration: float,
    warmup: float,
    pulse_value: float,
    pulse_duration: float,
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

    X_pulse = pulse_func(pulse_value, pulse_duration, warmup - pulse_duration)

    timer.start()
    t = 0
    while t < duration:
        t = timer.elapsed_time()
        rq, rdq, rpa, rpb = state.get_raw_states()
        q, dq, pa, pb = state.get_states()
        qdes = qdes_func(t)
        dqdes = dqdes_func(t)
        ca, cb = ctrl.update(t, q, dq, pa, pb, qdes, dqdes)
        X = np.atleast_2d(X_pulse(t))
        y = np.ravel(model.predict(X))
        if t > warmup:
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


def mainloop(args: argparse.Namespace, model: reBASICS | None = None):
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
        # qdes_func = spline.get_qdes_func(q0)
        # dqdes_func = spline.get_dqdes_func(q0)
        if model is None:
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
            control_reBASICS(
                model,
                comm,
                ctrl,
                state,
                args.joint,
                qdes_func,
                dqdes_func,
                args.duration,
                args.warmup,
                args.pulse_value,
                args.pulse_duration,
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
        choices=["pid", "rebasics"],
        default="rebasics",
        help="Controller type to be executed.",
    )
    parser.add_argument(
        "--train-data",
        required=True,
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
        default=100.0,
        type=float,
        help="sensor frequency",
    )
    parser.add_argument(
        "-f",
        "--control-freq",
        dest="cfreq",
        default=30.0,
        type=float,
        help="control frequency",
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
        "-v",
        "--pulse-value",
        default=5.0,
        type=float,
        help="Value of input pulse",
    )
    parser.add_argument(
        "-p",
        "--pulse-duration",
        default=0.05,
        type=float,
        help="Duration of input pulse",
    )
    parser.add_argument(
        "-w",
        "--warmup",
        default=0.25,
        type=float,
        help="Duration of warming up",
    )
    parser.add_argument(
        "-n",
        "--n-train-loop",
        default=10,
        type=int,
        help="Loop number of training",
    )
    parser.add_argument(
        "--n-neurons",
        default=100,
        type=int,
        help="The number of neurons in each reservoir.",
    )
    parser.add_argument(
        "--n-modules",
        default=800,
        type=int,
        help="The number of reservoir modules.",
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
        default=1.2,
        type=float,
        help="Spectral radius of the inter connection weight matrix.",
    )
    parser.add_argument(
        "--leaking-rate",
        default=0.1,
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
    parser.add_argument("-m", "--model", help="path to saved model")
    return parser.parse_args()


def convert_args_to_reBASICS_params(args):
    args_dict = vars(args).copy()
    keys = [
        "n_neurons",
        "n_modules",
        "density",
        "input_scale",
        "rho",
        "leaking_rate",
        "noise_level",
    ]
    params = {k: args_dict[k] for k in keys}
    params["N_x"] = params.pop("n_neurons")
    params["N_module"] = params.pop("n_modules")
    return params


def main():
    args = parse()
    params = convert_args_to_reBASICS_params(args)
    sfparam = convert_args_to_sfparam(args)
    if args.ctrl.lower() == "rebasics":
        loader = Loader(args.joint, args.n_predict, args.n_ctrl_period)
        X_train, y_train = loader.load(args.train_data)
        if args.model is None:
            model = fit(args, X_train, y_train, params)
            joblib.dump(model, "model.joblib")
        else:
            model = joblib.load(args.model)
        if args.test_data is not None:
            X_test, y_test = loader.load(args.test_data)
            plot(args, model, X_test, y_test, sfparam)
        else:
            mainloop(args, model)
    else:
        mainloop(args, None)


if __name__ == "__main__":
    main()
