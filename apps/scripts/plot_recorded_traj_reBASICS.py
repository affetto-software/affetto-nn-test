#!/usr/bin/env python

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from pyplotutil.datautil import Data

sfparam_tmpl = {
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


def plot_command(data, data_ref, joints, warmup=0.0, **sfparam):
    mask = data.t > warmup
    for i in joints:
        fig, ax = plt.subplots()
        title = f"pressure at valve (joint={i:02})"
        (line,) = ax.plot(
            data_ref.t,
            getattr(data_ref, f"ca{i}") * 600 / 255,
            ls="--",
            label=f"reference",
        )
        ax.plot(
            data.t[mask] - warmup,
            getattr(data, f"ca{i}")[mask] * 600 / 255,
            c=line.get_color(),
            ls="-",
            label=f"actual",
        )
        (line,) = ax.plot(
            data_ref.t,
            getattr(data_ref, f"cb{i}") * 600 / 255,
            ls="--",
            label=f"reference",
        )
        ax.plot(
            data.t[mask] - warmup,
            getattr(data, f"cb{i}")[mask] * 600 / 255,
            c=line.get_color(),
            ls="-",
            label=f"actual",
        )
        ax.grid(axis="y")
        ax.legend()
        pparam = {
            "title": title,
            "xlabel": "time [s]",
            "ylabel": "command [0-600]",
            "ylim": [0, 600],
        }
        ax.set(**pparam)
        sfparam["filename"] = (
            title.translate(
                str.maketrans({" ": "_", "=": "-", "(": None, ")": None, ",": None})
            )
            + ".svg"
        )
        savefig(fig, **sfparam)


def plot_pressure(data, data_ref, joints, warmup=0.0, **sfparam):
    mask = data.t > warmup
    for i in joints:
        fig, ax = plt.subplots()
        title = f"pressure at actuator (joint={i:02})"
        (line,) = ax.plot(
            data_ref.t,
            getattr(data_ref, f"pa{i}"),
            ls="--",
            label=f"reference",
        )
        ax.plot(
            data.t[mask] - warmup,
            getattr(data, f"pa{i}")[mask],
            c=line.get_color(),
            ls="-",
            label=f"actual",
        )
        (line,) = ax.plot(
            data_ref.t,
            getattr(data_ref, f"pb{i}"),
            ls="--",
            label=f"reference",
        )
        ax.plot(
            data.t[mask] - warmup,
            getattr(data, f"pb{i}")[mask],
            c=line.get_color(),
            ls="-",
            label=f"actual",
        )
        ax.grid(axis="y")
        ax.legend()
        pparam = {
            "title": title,
            "xlabel": "time [s]",
            "ylabel": "pressure [0-600]",
            "ylim": [0, 600],
        }
        ax.set(**pparam)
        sfparam["filename"] = (
            title.translate(
                str.maketrans({" ": "_", "=": "-", "(": None, ")": None, ",": None})
            )
            + ".svg"
        )
        savefig(fig, **sfparam)


def plot_q(data, data_ref, joints, warmup=0.0, **sfparam):
    mask = data.t > warmup
    for i in joints:
        fig, ax = plt.subplots()
        title = f"joint angle (joint={i:02})"
        (line,) = ax.plot(
            data_ref.t,
            getattr(data_ref, f"q{i}"),
            ls="--",
            label=f"reference",
        )
        ax.plot(
            data.t[mask] - warmup,
            getattr(data, f"q{i}")[mask],
            # c=line.get_color(),
            ls="-",
            label=f"actual",
        )
        ax.grid(axis="y")
        ax.legend()
        pparam = {
            "title": title,
            "xlabel": "time [s]",
            "ylabel": "angle [0-100]",
        }
        ax.set(**pparam)
        sfparam["filename"] = (
            title.translate(
                str.maketrans({" ": "_", "=": "-", "(": None, ")": None, ",": None})
            )
            + ".svg"
        )
        savefig(fig, **sfparam)


def plot_dq(data, data_ref, joints, warmup=0.0, **sfparam):
    mask = data.t > warmup
    for i in joints:
        fig, ax = plt.subplots()
        title = f"joint angle velocity (joint={i:02})"
        (line,) = ax.plot(
            data_ref.t,
            getattr(data_ref, f"dq{i}"),
            ls="--",
            label=f"reference",
        )
        ax.plot(
            data.t[mask] - warmup,
            getattr(data, f"dq{i}")[mask],
            # c=line.get_color(),
            ls="-",
            label=f"actual",
        )
        ax.grid(axis="y")
        ax.legend()
        pparam = {
            "title": title,
            "xlabel": "time [s]",
            "ylabel": "angle velocity [/s]",
        }
        ax.set(**pparam)
        sfparam["filename"] = (
            title.translate(
                str.maketrans({" ": "_", "=": "-", "(": None, ")": None, ",": None})
            )
            + ".svg"
        )
        savefig(fig, **sfparam)


def parse():
    parser = argparse.ArgumentParser(
        description="Plot script for send_sinusoidal_command.py"
    )
    parser.add_argument("data", help="path to data file")
    parser.add_argument("-r", "--reference", required=True, help="reference data file")
    parser.add_argument(
        "-j",
        "--joints",
        nargs="+",
        required=True,
        type=int,
        help="joint indices to be shown in figure",
    )
    parser.add_argument(
        "-w",
        "--warmup",
        default=0.25,
        type=float,
        help="Duration of warming up",
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
        "-s", "--savefig", action="store_true", help="export figures if specified"
    )
    parser.add_argument(
        "-x", "--noshow", action="store_true", help="do not show figures if specified"
    )
    return parser.parse_args()


def main():
    args = parse()
    sfparam = sfparam_tmpl.copy()
    sfparam["savefig"] = args.savefig
    sfparam["basedir"] = args.basedir
    sfparam["extensions"] = args.extension

    data = Data(args.data)
    reference = Data(args.reference)
    plot_command(data, reference, args.joints, warmup=args.warmup, **sfparam)
    plot_pressure(data, reference, args.joints, warmup=args.warmup, **sfparam)
    plot_q(data, reference, args.joints, warmup=args.warmup, **sfparam)
    plot_dq(data, reference, args.joints, warmup=args.warmup, **sfparam)

    if not args.noshow:
        plt.show()


if __name__ == "__main__":
    main()
