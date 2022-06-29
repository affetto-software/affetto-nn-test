from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline


def plot_prediction(
    reg: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    index: int | Iterable[int] = 0,
    title: str | None = None,
) -> None:
    y_predict = reg.predict(X_test)
    _, ax = plt.subplots()
    if not isinstance(index, Iterable):
        index = [index]
    for i in index:
        ax.plot(y_test[:, i], label="test")
        ax.plot(y_predict[:, i], label="predict")
    if title is not None:
        ax.set_title(title)
    ax.legend()
