#!/usr/bin/env python

import argparse
import itertools
import time
from pathlib import Path
from typing import Callable

import numpy as np

from affctrllib import AffComm, AffPosCtrl, AffStateThread, Logger, Timer

DEFAULT_CONFIG_PATH = Path(__file__).parent.joinpath("config.toml")
DEFAULT_JOINT_LIST = [0]
DEFAULT_DURATION = 20.0  # sec
DEFAULT_AMPLITUDE_LIST = [40.0]
DEFAULT_PERIOD_LIST = [5.0, 10.0]
DEFAULT_BIAS_LIST = [50.0]
DEFAULT_N_REPEAT = 10


def prepare_ctrl(
    config: str, sfreq: float | None, cfreq: float | None
) -> tuple[AffComm, AffPosCtrl, AffStateThread]:
    comm = AffComm(config_path=config)
    comm.create_command_socket()
    state = AffStateThread(
        config=config, freq=sfreq, logging=False, output=None, butterworth=True
    )
    ctrl = AffPosCtrl(config_path=config, freq=cfreq)
    state.prepare()
    state.start()
    print("Waiting until robot gets stationary...")
    time.sleep(3)
    return comm, ctrl, state


def prepare_output_directory(output_directory: str | Path) -> Path:
    if output_directory is None:
        raise RuntimeError("Output directory is required.")
    path = Path(output_directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_logger(dof: int) -> Logger:
    logger = Logger()
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
        q[joint] = A * np.sin(2.0 * np.pi * t / T) + b
        return q

    def dqdes_func(t: float) -> np.ndarray:
        dq = np.zeros((len(q0),))
        dq[joint] = A * T * np.cos(2.0 * np.pi * t / T) / (2.0 * np.pi)
        return dq

    return qdes_func, dqdes_func


def create_step_trajectory(
    A: float, T: float, b: float, joint: int, q0: np.ndarray
) -> tuple[Callable[[float], np.ndarray], Callable[[float], np.ndarray]]:
    def qdes_func(t: float) -> np.ndarray:
        q = np.copy(q0)
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
        logger.set_filename(
            log_filename if log_filename is not None else Path("data.csv")
        )
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
        timer.block()
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


def record(
    comm: AffComm,
    ctrl: AffPosCtrl,
    state: AffStateThread,
    output_dir: Path,
    logger: Logger,
    q0: np.ndarray,
    traj_type: str,
    joint: int,
    duration: float,
    A: float,
    T: float,
    b: float,
    i: int,
    cnt: int,
    N: int,
):
    print("Preparing for next trajectory...")
    qdes_func, dqdes_func = TRAJECTORY["const"](A, T, b, joint, q0)
    control(comm, ctrl, state, qdes_func, dqdes_func, 3)
    print(f"Recording {cnt + 1}/{N} (joint={joint}, A={A}, T={T}, b={b}, i={i})...")
    qdes_func, dqdes_func = TRAJECTORY[traj_type](A, T, b, joint, q0)
    log_filename = (
        output_dir / f"{traj_type}_joint-{joint}_A-{A}_T-{T}_b-{b}_{i:02}.csv"
    )
    control(comm, ctrl, state, qdes_func, dqdes_func, duration, logger, log_filename)


def mainloop(
    config: str,
    output: str,
    traj_type: str,
    joint: int,
    duration: float,
    amplitude: list[float],
    period: list[float],
    bias: list[float],
    n_repeat: int,
    sfreq: float | None = None,
    cfreq: float | None = None,
):
    # Prepare controller.
    comm, ctrl, state = prepare_ctrl(config, sfreq, cfreq)

    # Prepare output directory.
    output_dir = prepare_output_directory(output)

    # Prepare logger.
    logger = prepare_logger(ctrl.dof)

    # Get initial pose.
    q0 = state.q

    # Record trajectories.
    N = len(amplitude) * len(period) * len(bias) * n_repeat
    cnt = 0
    try:
        for A, T, b in itertools.product(amplitude, period, bias):
            for i in range(n_repeat):
                record(
                    comm,
                    ctrl,
                    state,
                    output_dir,
                    logger,
                    q0,
                    traj_type,
                    joint,
                    duration,
                    A,
                    T,
                    b,
                    i,
                    cnt,
                    N,
                )
                cnt += 1
    finally:
        print("Quitting...")
        comm.close_command_socket()
        state.join()


def parse():
    parser = argparse.ArgumentParser(
        description="Generate data by providing periodic positional trajectory."
    )
    parser.add_argument(
        "-c",
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Config file path for control.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output directory where collected data files are stored.",
    )
    parser.add_argument(
        "-F",
        "--sensor-freq",
        dest="sfreq",
        type=float,
        help="Sensor frequency.",
    )
    parser.add_argument(
        "-f",
        "--control-freq",
        dest="cfreq",
        type=float,
        help="Control frequency.",
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
        "-j",
        "--joint",
        nargs="+",
        default=DEFAULT_JOINT_LIST,
        type=int,
        help="Joint index (list) to move.",
    )
    parser.add_argument(
        "-T",
        "--time",
        default=DEFAULT_DURATION,
        type=float,
        help="Time duration for each trajectory.",
    )
    parser.add_argument(
        "-a",
        "--amplitude",
        nargs="+",
        default=DEFAULT_AMPLITUDE_LIST,
        type=float,
        help="Amplitude list for reference trajectory.",
    )
    parser.add_argument(
        "-p",
        "--period",
        nargs="+",
        default=DEFAULT_PERIOD_LIST,
        type=float,
        help="Period list for reference trajectory.",
    )
    parser.add_argument(
        "-b",
        "--bias",
        nargs="+",
        default=DEFAULT_BIAS_LIST,
        type=float,
        help="Bias list for reference trajectory.",
    )
    parser.add_argument(
        "-n",
        "--n-repeat",
        dest="n",
        default=DEFAULT_N_REPEAT,
        type=int,
        help="Number of times to repeat each reference trajectory.",
    )
    return parser.parse_args()


def main():
    args = parse()
    mainloop(
        args.config,
        args.output,
        args.traj_type,
        args.joint[0],
        args.time,
        args.amplitude,
        args.period,
        args.bias,
        args.n,
        args.sfreq,
        args.cfreq,
    )


if __name__ == "__main__":
    main()
