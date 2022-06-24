#!/usr/bin/env python

import argparse
import itertools
import time
from pathlib import Path
from typing import Callable

import numpy as np
from affctrllib import AffPosCtrlThread, Logger, Timer

DEFAULT_CONFIG_PATH = Path(__file__).parent.joinpath("config.toml")
DEFAULT_LOGGING_FREQ = 10
DEFAULT_DURATION = 20  # sec
# AMPLITUDE_LIST = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
AMPLITUDE_LIST = [40]
PERIOD_LIST = [5, 10]
BIAS_LIST = [50]
REPEAT_N = 1


def prepare_ctrl(
    config: str, sfreq: float | None, cfreq: float | None
) -> AffPosCtrlThread:
    ctrl = AffPosCtrlThread(config=config, freq=cfreq, sensor_freq=sfreq)
    ctrl.start()
    ctrl.wait_for_idling()
    print("Waiting until robot gets stationary...")
    time.sleep(5)
    return ctrl


def prepare_output(output: str) -> Path:
    if output is None:
        raise RuntimeError("Output directory is required")
    path = Path(output) / "data"
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


def get_qdes_trajectory(
    A: float, T: float, b: float, joint: int, q0: np.ndarray
) -> Callable[[float], np.ndarray]:
    def qdes_func(t: float) -> np.ndarray:
        q = np.copy(q0)
        if t - int(t / T) * T < 0.5 * T:
            q[joint] = b + A
        else:
            q[joint] = b - A
        return q

    return qdes_func


def get_dqdes_trajectory(q0: np.ndarray) -> Callable[[float], np.ndarray]:
    def dqdes_func(_: float) -> np.ndarray:
        dq = np.zeros((len(q0),))
        return dq

    return dqdes_func


def prepare_for_next_trajectory(
    ctrl: AffPosCtrlThread, q0: np.ndarray, joint: int
) -> None:
    q = q0.copy()
    q[joint] = 50
    ctrl.reset_trajectory(q)
    time.sleep(3)


def record_one_trajectory(
    ctrl: AffPosCtrlThread,
    logger: Logger,
    log_filename: str | Path,
    lfreq: float | None,
    qdes_func: Callable[[float], np.ndarray],
    dqdes_func: Callable[[float], np.ndarray],
    duration: float,
) -> None:
    timer = Timer(rate=lfreq if lfreq is not None else DEFAULT_LOGGING_FREQ)
    logger.erase_data()
    logger.set_filename(log_filename)
    ctrl.reset_timer()
    ctrl.set_trajectory(qdes_func, dqdes_func)
    timer.start()
    t = 0
    time.sleep(2 * ctrl.dt)  # wait until controller updated at least once
    while t < duration:
        t = timer.elapsed_time()
        logger.store(ctrl.logger.get_data()[-1].copy())
        timer.block()
    logger.dump()


def generate_sin_trajectory(
    ctrl: AffPosCtrlThread,
    output_dir: Path,
    logger: Logger,
    joint: int,
    duration: float,
    lfreq: float,
    A: float,
    T: float,
    b: float,
    i: int,
) -> None:
    q0 = ctrl.state.q
    qdes_func = get_qdes_trajectory(A, T, b, joint, q0)
    dqdes_func = get_dqdes_trajectory(q0)
    log_filename = output_dir / f"step-joint-{joint}_A-{A}_T-{T}_b-{b}_{i:02}.csv"
    print("Preparing next trajectory...")
    prepare_for_next_trajectory(ctrl, q0, joint)
    print(f"Recording (joint={joint}, A={A}, T={T}, b={b}, i={i})...")
    record_one_trajectory(
        ctrl, logger, log_filename, lfreq, qdes_func, dqdes_func, duration
    )


def mainloop(
    config: str,
    output: str,
    joint: int = 0,
    duration: float = 10,
    sfreq: float | None = None,
    cfreq: float | None = None,
    lfreq: float | None = None,
):
    # Start AffPosCtrlThread
    ctrl = prepare_ctrl(config, sfreq, cfreq)

    # Prepare output directory.
    output_dir = prepare_output(output)

    # Create logger.
    logger = prepare_logger(ctrl.dof)

    # Record trajectories.
    try:
        for (A, T, b) in itertools.product(AMPLITUDE_LIST, PERIOD_LIST, BIAS_LIST):
            for i in range(REPEAT_N):
                generate_sin_trajectory(
                    ctrl, output_dir, logger, joint, duration, lfreq, A, T, b, i
                )
    finally:
        print("Quitting...")
        ctrl.join()


def parse():
    parser = argparse.ArgumentParser(
        description="Generate data by providing sinusoidal positional trajectory"
    )
    parser.add_argument(
        "-c", "--config", default=DEFAULT_CONFIG_PATH, help="config file"
    )
    parser.add_argument("-o", "--output", required=True, help="output directory")
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
        "-L",
        "--logging-freq",
        dest="lfreq",
        type=float,
        help="logging frequency",
    )
    parser.add_argument(
        "-j", "--joint", default=0, type=int, help="Joint index to move"
    )
    parser.add_argument(
        "-T",
        "--time",
        default=DEFAULT_DURATION,
        type=float,
        help="Time duration for each recording",
    )
    return parser.parse_args()


def main():
    args = parse()
    mainloop(
        args.config,
        args.output,
        args.joint,
        args.time,
        args.sfreq,
        args.cfreq,
        args.lfreq,
    )


if __name__ == "__main__":
    main()
