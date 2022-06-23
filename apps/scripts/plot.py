#!/usr/bin/env python

import argparse
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


def plot_q(data_list, joint, **sfparam):
    fig, ax = plt.subplots()
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
    }
    ax.set(**pparam)
    if sfparam.get("filename", None) is None:
        sfparam["filename"] = "q"
    savefig(fig, **sfparam)


def plot_dq(data_list, joint, **sfparam):
    fig, ax = plt.subplots()
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
    }
    ax.set(**pparam)
    if sfparam.get("filename", None) is None:
        sfparam["filename"] = "dq"
    savefig(fig, **sfparam)


def plot_pressure(data_list, joint, **sfparam):
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
    }
    ax.set(**pparam)
    if sfparam.get("filename", None) is None:
        sfparam["filename"] = "pressure"
    savefig(fig, **sfparam)


def plot(data_dir: str | Path, joints: list[int], sfparam: dict[Any]):
    data_dir = Path(data_dir)
    for joint in joints:
        data_files = data_dir.glob(f"joint-{joint}_A-40_T-10_b-50_*.csv")
        data_list = [Data(f) for f in data_files]
        plot_q(data_list, joint, **sfparam)
        plot_dq(data_list, joint, **sfparam)
        plot_pressure(data_list, joint, **sfparam)


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
        "-t", "--time", nargs="+", type=float, help="time range to show in figure"
    )
    parser.add_argument(
        "-s", "--savefig", action="store_true", help="export figures if specified"
    )
    parser.add_argument(
        "-x", "--noshow", action="store_true", help="do not show figures if specified"
    )
    return parser.parse_args()


def main():
    args = parse()
    sfparam = sfparam_tmpl.copy()
    sfparam["time"] = args.time
    sfparam["savefig"] = args.savefig
    sfparam["basedir"] = args.basedir
    sfparam["extensions"] = args.extension

    plot(args.data_dir, args.joint, sfparam)
    if not args.noshow:
        plt.show()


if __name__ == "__main__":
    main()
