import pandas as pd
from typing import Callable, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def summarise(df: pd.DataFrame, columns: Optional[List[str]] = None) -> str:
    """
    Retrieves statistical information of every column of the provided dataframe.
    To summarise only selected columns, provide the columns names.
    :param df: dataframe to summarise.
    :param columns: optional list of column names to summarise specific columns.
    :return: printable string containing summary statistics.
    """
    if columns:
        df = df[:, columns]
    s = f"Summary statistics:\nDataframe dimensions: {len(df)}x{len(df.columns)}"
    for i in range(len(df.columns)):
        col = df.iloc[:, i]
        t = str(col.dtype)
        s += f"\n{col.name} - {t}"
        if t == "bool" or ((col == 0) | (col == 1)).all():
            col = col.astype("bool")
            counts = col.value_counts()
            s += f"\n\tCounts:"
            s += f"\n\t\tTrue: {counts[0]}"
            s += f"\n\t\tFalse: {counts[1]}"
        elif t in {"float64", "int64"}:
            s += f"\n\tRange({np.nanmin(col): .2f}, {np.nanmax(col): .2f})"
            col_q = col.quantile((.25, .5, .75)).values
            s += f"\n\tQuantiles: 0.25: {col_q[0]: 0.2f} | 0.5: {col_q[1]: 0.2f} | 0.75: {col_q[2]: 0.2f}"
            s += f"\n\tMean: {np.nanmean(col): .2f}"
        else:
            s += f"\n\tExamples:"
            for idx in range(3):
                s += f"\n\t\t{col[idx]}"
            s += f"\n\tUnique objects:"
            s += f"\n\t\t{len(col.unique())}"
        s += f"\n\tContains NaN: {pd.isna(col).any()}"
    return s



class plot:
    xs: np.ndarray
    history: list = []

    def __init__(self, from_a: float = -1, to_b: float = 1, num_samples: int = 100):
        self.init(from_a, to_b, num_samples)

    @staticmethod
    def init(from_a: float = -1, to_b: float = 1, num_samples: int = 100):
        plot.xs = np.linspace(from_a, to_b, num_samples)
        plot.history = [
            "import numpy as np",
            "import matplotlib.pyplot as plt\n",
            f"np.linspace({from_a}, {to_b}, {num_samples})"
        ]
        return plot

    @staticmethod
    def imshow(img: np.ndarray, **kwargs):
        if len(img.shape) == 2:
            kwargs["cmap"] = "gray"
        plt.imshow(img, **kwargs)
        plot.history.append(f'plt.imshow(img{plot._pargs([], **kwargs)})')
        return plot

    @staticmethod
    def plot(y, *args, **kwargs):
        plt.plot(plot.xs, y, *args, **kwargs)
        plot.history.append(f"plt.plot(xs, y{plot._pargs(*args, **kwargs)})")
        return plot

    @staticmethod
    def function(func, label: str = "", **kwargs) -> plt:
        plt.plot(plot.xs, func(plot.xs), label=label, **kwargs)
        plot.history.append(f"plt.plot(xs, func(xs){plot._pargs(label=label, **kwargs)})")
        return plot

    @staticmethod
    def functions(funcs: List[Callable], labels: List[str] = None):
        if labels is None:
            labels = [""] * len(funcs)

        for func, label in zip(funcs, labels):
            plot.function(func, label)
        return plot

    @staticmethod
    def data(data: dict, x: str, y: str, *args, **kwargs):
        return (plot
                .plot(x, y, *args, data=data, **kwargs)
                .xlabel(x)
                .ylabel(y)
                )

    @staticmethod
    def legend(*args, **kwargs):
        plt.legend(*args, **kwargs)
        plot.history.append(f"plt.legend({plot._pargs(*args, **kwargs)})")
        return plot

    @staticmethod
    def xlabel(label: str):
        plt.xlabel(label)
        plot.history.append(f'plt.xlabel("{label}")')
        return plot

    @staticmethod
    def ylabel(label: str):
        plt.ylabel(label)
        plot.history.append(f'plt.ylabel("{label}")')
        return plot

    @staticmethod
    def labels(title: str, xlabel: str, ylabel: str):
        plot.title(title)
        plot.xlabel(xlabel)
        plot.ylabel(ylabel)
        return plot

    @staticmethod
    def title(title: str):
        plt.title(title)
        plot.history.append(f'plt.title("{title}")')
        return plot

    @staticmethod
    def noaxis():
        plt.axis("off")
        plot.history.append('plt.axis("off")')
        return plot

    @staticmethod
    def show(*args, **kwargs):
        plot.history.append(f"plt.show({plot._pargs(*args, **kwargs)})")
        plt.show(*args, **kwargs)
        plt.history = []
        return plot

    @staticmethod
    def _pargs(*args, **kwargs):
        sargs = ""
        if len(args):
            sargs = ", " + ", ".join([f'"{arg}"' if isinstance(arg, str) else str(arg) for arg in args])

        skwargs = ""
        if len(kwargs):
            for key, value in kwargs.items():
                if isinstance(value, str):
                    value = f'"{value}"'
                skwargs += f", {key}={value}"

        return sargs + skwargs

    @staticmethod
    def export():
        return "\n".join(plot.history)

