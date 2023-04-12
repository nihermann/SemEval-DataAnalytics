import pandas as pd
from typing import Callable, List, Optional

import matplotlib.pyplot as plt
import numpy as np





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

