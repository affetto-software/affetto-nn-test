#!/usr/bin/env python

import argparse
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from pyplotutil.datautil import Data

sfparam_tmpl = {
    "time": None,
    "savefig": False,
    "basedir": "fig",
    "filename": None,
    "extensions": ["svg"],
}


def savefig(fig, **sfparam):
    if not sfparam.get("savefig", False):
        return
    if not "filename" in sfparam:
        raise KeyError(f"filename is required in sfparam.")

    # Join basedir and filename.
    path = Path(sfparam.get("basedir", "fig")) / Path(sfparam["filename"])
    # Create directories if needed.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Save figures in specified formats.
    for ext in sfparam.get("extensions", ["svg"]):
        if not ext.startswith("."):
            ext = f".{ext}"
        fname = path.with_suffix(ext)
        fig.savefig(str(fname), bbox_inches="tight")


def make_mask(t, between=None):
    if between is None:
        return np.full(t.size, True)
    elif len(between) == 1:
        return t <= between[0]
    else:
        return (t >= between[0]) & (t <= between[1])


def plot_q(data_list, joint, title, **sfparam):
    fig, ax = plt.subplots()
    if len(data_list) > 0:
        data = data_list[0]
        mask = make_mask(data.t, sfparam["time"])
        ax.plot(
            data.t[mask],
            getattr(data, f"qdes{joint}")[mask],
            label=f"qdes[{joint}]",
            lw=1,
        )
    for data in data_list:
        mask = make_mask(data.t, sfparam["time"])
        ax.plot(
            data.t[mask], getattr(data, f"q{joint}")[mask], label=f"q[{joint}]", lw=1
        )
    ax.grid(axis="y")
    # ax.legend(title="Joint angle")
    pparam = {
        "xlabel": "time [s]",
        "ylabel": "position [0-100]",
        "title": title,
    }
    ax.set(**pparam)
    if sfparam.get("filename", None) is None:
        sfparam["filename"] = title.translate(
            str.maketrans({" ": "_", "=": "-", "(": None, ")": None, ",": None})
        )
    savefig(fig, **sfparam)


def plot_dq(data_list, joint, title, **sfparam):
    fig, ax = plt.subplots()
    if len(data_list) > 0:
        data = data_list[0]
        mask = make_mask(data.t, sfparam["time"])
        ax.plot(
            data.t[mask],
            getattr(data, f"dqdes{joint}")[mask],
            label=f"dqdes[{joint}]",
            lw=1,
        )
    for data in data_list:
        mask = make_mask(data.t, sfparam["time"])
        ax.plot(
            data.t[mask], getattr(data, f"dq{joint}")[mask], label=f"dq[{joint}]", lw=1
        )
    ax.grid(axis="y")
    # ax.legend(title="Joint velocity")
    pparam = {
        "xlabel": "time [s]",
        "ylabel": "velocity [/s]",
        "title": title,
    }
    ax.set(**pparam)
    if sfparam.get("filename", None) is None:
        sfparam["filename"] = title.translate(
            str.maketrans({" ": "_", "=": "-", "(": None, ")": None, ",": None})
        )
    savefig(fig, **sfparam)


def plot_pressure(data_list, joint, title, **sfparam):
    fig, ax = plt.subplots()
    for data in data_list:
        mask = make_mask(data.t, sfparam["time"])
        ax.plot(
            data.t[mask], getattr(data, f"pa{joint}")[mask], label=f"pa[{joint}]", lw=1
        )
        ax.plot(
            data.t[mask], getattr(data, f"pb{joint}")[mask], label=f"pb[{joint}]", lw=1
        )
    ax.grid(axis="y")
    # ax.legend(title="Measured pressure")
    pparam = {
        "xlabel": "time [s]",
        "ylabel": "pressure [kPa]",
        "title": title,
    }
    ax.set(**pparam)
    if sfparam.get("filename", None) is None:
        sfparam["filename"] = title.translate(
            str.maketrans({" ": "_", "=": "-", "(": None, ")": None, ",": None})
        )
    savefig(fig, **sfparam)


def plot_pressure_valve(data_list, joint, title, **sfparam):
    fig, ax = plt.subplots()
    for data in data_list:
        mask = make_mask(data.t, sfparam["time"])
        ax.plot(
            data.t[mask],
            getattr(data, f"ca{joint}")[mask] * 600 / 255,
            label=f"ca[{joint}]",
            lw=1,
        )
        ax.plot(
            data.t[mask],
            getattr(data, f"cb{joint}")[mask] * 600 / 255,
            label=f"cb[{joint}]",
            lw=1,
        )
    ax.grid(axis="y")
    # ax.legend(title="Measured pressure")
    pparam = {
        "xlabel": "time [s]",
        "ylabel": "pressure [kPa]",
        "title": title,
    }
    ax.set(**pparam)
    if sfparam.get("filename", None) is None:
        sfparam["filename"] = "pressure_valve"
        sfparam["filename"] = title.translate(
            str.maketrans({" ": "_", "=": "-", "(": None, ")": None, ",": None})
        )
    savefig(fig, **sfparam)


def plot(
    data_dir: str | Path,
    joints: list[int],
    traj_type: str,
    A: float,
    T: float,
    b: float,
    sfparam: dict[str, Any],
):
    data_dir = Path(data_dir)
    for j in joints:
        # pattern = f"{traj_type}[-_]joint-{j}_A-{A}_T-{T}_b-{b}_*.csv"
        pattern = (
            f"{traj_type}[-_]joint-{j}_A-{round(A)}_T-{round(T)}_b-{round(b)}_*.csv"
        )
        data_files = data_dir.glob(pattern)
        data_list = [Data(f) for f in data_files]
        if len(data_list) == 0:
            warnings.warn(
                f"No data files found in {str(data_dir)} (pattern: {pattern})"
            )
        suffix = f" (A={A}, T={T}, b={b})"
        plot_q(data_list, j, "Joint angle" + suffix, **sfparam)
        plot_dq(data_list, j, "Joint velocity" + suffix, **sfparam)
        plot_pressure(data_list, j, "Pressure at actuator" + suffix, **sfparam)
        plot_pressure_valve(data_list, j, "Pressure at valve" + suffix, **sfparam)


def parse():
    parser = argparse.ArgumentParser(description="Plot script for testing")
    parser.add_argument("data_dir", help="path to data directory")
    parser.add_argument(
        "-j",
        "--joint",
        nargs="+",
        type=int,
        help="joint indices to be shown in figure",
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
    parser.add_argument(
        "-t",
        "--trajectory",
        dest="traj_type",
        default="sin",
        choices=["sin", "step"],
        help="Trajectory type to be generated.",
    )
    parser.add_argument(
        "-a",
        "--amplitude",
        default=40.0,
        type=float,
        help="Amplitude for reference trajectory.",
    )
    parser.add_argument(
        "-p",
        "--period",
        default=5.0,
        type=float,
        help="Period for reference trajectory.",
    )
    parser.add_argument(
        "-b",
        "--bias",
        default=50.0,
        type=float,
        help="Bias for reference trajectory.",
    )
    return parser.parse_args()


def main():
    args = parse()
    sfparam = sfparam_tmpl.copy()
    sfparam["time"] = args.time
    sfparam["savefig"] = args.savefig
    sfparam["basedir"] = args.basedir
    sfparam["extensions"] = args.extension

    plot(
        args.data_dir,
        args.joint,
        args.traj_type,
        args.amplitude,
        args.period,
        args.bias,
        sfparam,
    )
    if not args.noshow:
        plt.show()


if __name__ == "__main__":
    main()
