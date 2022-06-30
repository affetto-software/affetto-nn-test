import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline

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


def convert_args_to_sfparam(args: argparse.Namespace) -> dict[str, Any]:
    sfparam = sfparam_tmpl.copy()
    sfparam["time"] = args.time
    sfparam["savefig"] = args.savefig
    sfparam["basedir"] = args.basedir
    sfparam["extensions"] = args.extension
    return sfparam


def plot_prediction(
    reg: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    index: int | Iterable[int] = 0,
    title: str | None = None,
    **sfparam,
) -> None:
    y_predict = reg.predict(X_test)
    fig, ax = plt.subplots()
    if not isinstance(index, Iterable):
        index = [index]
    for i in index:
        ax.plot(y_test[:, i], label="test")
        ax.plot(y_predict[:, i], label="predict")
    if title is not None:
        ax.set_title(title)
    ax.legend()
    if sfparam.get("filename", None) is None:
        if title is None:
            sfparam["filename"] = "plot"
        else:
            sfparam["filename"] = title.translate(
                str.maketrans({" ": "_", "=": "-", "(": None, ")": None, ",": None})
            )
    savefig(fig, **sfparam)
