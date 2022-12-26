#!/usr/bin/env python

import argparse
import random
import time
from pathlib import Path
from typing import Callable, Generator, Iterable

import numpy as np
import pandas as pd
from affctrllib import AffComm, AffPosCtrl, AffStateThread, Logger, Timer

DEFAULT_CONFIG_PATH = Path(__file__).parent.joinpath("config.toml")
DEFAULT_JOINT_LIST = [0]
DEFAULT_DURATION = 60.0  # sec
DEFAULT_SEED = None
DEFAULT_MAX_UPDATE_DIFF = 40.0
DEFAULT_TIME_UPDATE = 1.0
DEFAULT_LIMIT = [5.0, 95.0]
DEFAULT_N_REPEAT = 10


GeneratorT = Generator[
    tuple[Callable[[float], np.ndarray], Callable[[float], np.ndarray]],
    None,
    None,
]


def prepare_ctrl(
    config: str, sfreq: float | None, cfreq: float | None
) -> tuple[AffComm, AffPosCtrl, AffStateThread]:
    comm = AffComm(config_path=config)
    comm.create_command_socket()
    state = AffStateThread(config=config, freq=sfreq, logging=False, output=None)
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
        # coordinates of the color blob
        ["x", "y", "z", "r"],
    )
    return logger


def create_const_trajectory(
    qdes: float | list[float] | np.ndarray, joint: int | list[int], q0: np.ndarray
) -> tuple[Callable[[float], np.ndarray], Callable[[float], np.ndarray]]:
    def qdes_func(_: float) -> np.ndarray:
        q = np.copy(q0)
        q[0] = 50  # make waist joint keep at middle.
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
):
    timer = Timer(rate=ctrl.freq)
    timer.start()
    t = 0
    while t < duration:
        t = timer.elapsed_time()
        # rq, rdq, rpa, rpb = state.get_raw_states()
        q, dq, pa, pb = state.get_states()
        qdes = qdes_func(t)
        dqdes = dqdes_func(t)
        ca, cb = ctrl.update(t, q, dq, pa, pb, qdes, dqdes)
        comm.send_commands(ca, cb)
        timer.block()


def _control_random(
    comm: AffComm,
    ctrl: AffPosCtrl,
    state: AffStateThread,
    timer: Timer,
    t0: float,
    T: float,
    qdes_func: Callable[[float], np.ndarray],
    dqdes_func: Callable[[float], np.ndarray],
) -> tuple[
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    t = t0
    rq, rdq, rpa, rpb = np.zeros(state.dof, dtype=float)
    q, dq, pa, pb = np.zeros(state.dof, dtype=float)
    qdes, dqdes, ca, cb = np.zeros(state.dof, dtype=float)

    while t < t0 + T:
        t = timer.elapsed_time()
        rq, rdq, rpa, rpb = state.get_raw_states()
        q, dq, pa, pb = state.get_states()
        qdes = qdes_func(t)
        dqdes = dqdes_func(t)
        ca, cb = ctrl.update(t, q, dq, pa, pb, qdes, dqdes)
        comm.send_commands(ca, cb)
        timer.block()
    return t, rq, rdq, rpa, rpb, q, dq, pa, pb, qdes, dqdes, ca, cb


def control_random(
    comm: AffComm,
    ctrl: AffPosCtrl,
    state: AffStateThread,
    joint: int | list[int],
    generator: GeneratorT,
    T: float,  # Time to update next pose of the robot
    duration: float,  # Time duration for one sequence of movements
    logger: Logger | None = None,
    log_filename: str | Path | None = None,
    quiet: bool = True,
):
    timer = Timer(rate=ctrl.freq)
    if logger is not None:
        logger.erase_data()
        logger.set_filename(
            log_filename if log_filename is not None else Path("data.csv")
        )

    timer.start()
    t = 0
    for qdes_func, dqdes_func in generator:
        if t > duration:
            break
        t0 = t

        # It taks 'T' sec to execute the function below. It returns
        # the last sensor states and commands. We expect that the
        # robot stands still, which means stops completely, at this
        # moment.
        t, rq, rdq, rpa, rpb, q, dq, pa, pb, qdes, dqdes, ca, cb = _control_random(
            comm, ctrl, state, timer, t0, T, qdes_func, dqdes_func
        )

        # Get x, y and z positions of color blobs from the camera.
        # Since coordinates.csv will be updated by other program,
        # update frequency of this 'for' loop must be synchronized
        # with updates of coordinates.csv.
        try:
            df = pd.read_csv("~/Desktop/erhan_cam/coordinates.csv")
            # Please check if tolist()[0] is correct. Should it be
            # tolist()[-1]?
            x, y, z, r = df.values.tolist()[0]
        # to prevent any failures just in case the csv file is written on in that moment
        except:
            x, y, z, r = 0, 0, 0, 0
            print("Couldnt read file 2")

        # Save sensor data in the logger.
        if logger is not None:
            logger.store(
                t, rq, rdq, rpa, rpb, q, dq, pa, pb, ca, cb, qdes, dqdes, x, y, z, r
            )

        if not quiet:
            print(
                f"T={T}, qdes={qdes_func(t)[joint]}, blob position({r})=[{x}, {y}, {z}]"
            )
        t = timer.elapsed_time()
    if logger is not None:
        logger.dump()


def random_trajectory_generator(
    q0: np.ndarray,
    joint: int | list[int],
    seed: int | None,
    qdes_first: float | list[float] | np.ndarray,
    diff_max: float,
    limit: list[float],
) -> GeneratorT:
    if len(limit) <= 1:
        ValueError(f"Size of limit requires at least 2: len(limit)={len(limit)}")
    elif limit[0] >= limit[1]:
        ValueError(f"First element must be lower then sencond: {limit[0]} < {limit[1]}")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    qdes = np.atleast_1d(qdes_first)
    while True:
        qdes_new = np.random.uniform(qdes - diff_max, qdes + diff_max)
        qdes_new[qdes_new < limit[0]] = limit[0] + (
            limit[0] - qdes_new[qdes_new < limit[0]]
        )
        qdes_new[qdes_new > limit[1]] = limit[1] - (
            qdes_new[qdes_new > limit[1]] - limit[1]
        )
        qdes_new = np.minimum(np.maximum(limit[0], qdes_new), limit[1])
        yield create_const_trajectory(qdes_new, joint, q0)
        qdes = qdes_new


def record(
    comm: AffComm,
    ctrl: AffPosCtrl,
    state: AffStateThread,
    output_dir: Path,
    logger: Logger,
    q0: np.ndarray,
    joint: int | list[int],
    duration: float,
    seed: int | None,
    diff_max: float,
    t_update: float,
    limit: list[float],
    i: int,
    cnt: int,
    N: int,
):
    print("Preparing for next trajectory...")
    if isinstance(joint, Iterable):
        qdes_first = [50.0 for _ in range(len(joint))]
    else:
        qdes_first = 50.0
    qdes_func, dqdes_func = create_const_trajectory(qdes_first, joint, q0)
    control(comm, ctrl, state, qdes_func, dqdes_func, 3)
    print(f"Recording {cnt + 1}/{N} (joint={joint}, i={i})...")
    joint_str = str(joint).strip("[]").replace(" ", "")
    log_filename = output_dir / f"random_joint-{joint_str}_{i:02}.csv"
    generator = random_trajectory_generator(
        q0, joint, seed, qdes_first, diff_max, limit
    )
    control_random(
        comm, ctrl, state, joint, generator, t_update, duration, logger, log_filename
    )


def mainloop(
    config: str,
    output: str,
    joint: int | list[int],
    duration: float,
    seed: int | None,
    diff_max: float,
    t_update: float,
    limit: list[float],
    n_repeat: int,
    start_index: int,
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
    cnt = 0
    try:
        for i in range(start_index, start_index + n_repeat):
            record(
                comm,
                ctrl,
                state,
                output_dir,
                logger,
                q0,
                joint,
                duration,
                seed,
                diff_max,
                t_update,
                limit,
                i,
                cnt,
                n_repeat,
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
        "-s",
        "--seed",
        default=DEFAULT_SEED,
        type=int,
        help="Seed value given to random number generator.",
    )
    parser.add_argument(
        "-d",
        "--max-diff",
        default=DEFAULT_MAX_UPDATE_DIFF,
        type=float,
        help="Maximum position difference when updating desired position.",
    )
    parser.add_argument(
        "-t",
        "--time-update",
        default=DEFAULT_TIME_UPDATE,
        type=float,
        help="Time to update the next desired position of the robot.",
    )
    parser.add_argument(
        "-l",
        "--limit",
        nargs="+",
        default=DEFAULT_LIMIT,
        type=float,
        help="Joint limit.",
    )
    parser.add_argument(
        "-n",
        "--n-repeat",
        dest="n",
        default=DEFAULT_N_REPEAT,
        type=int,
        help="Number of times to repeat each reference trajectory.",
    )
    parser.add_argument(
        "-i",
        "--start-index",
        default=0,
        type=int,
        help="Start index to record reference trajectory.",
    )
    return parser.parse_args()


def main():
    args = parse()
    mainloop(
        args.config,
        args.output,
        args.joint,
        args.time,
        args.seed,
        args.max_diff,
        args.time_update,
        args.limit,
        args.n,
        args.start_index,
        args.sfreq,
        args.cfreq,
    )


if __name__ == "__main__":
    main()
