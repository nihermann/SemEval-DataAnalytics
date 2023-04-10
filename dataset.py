from typing import List, Tuple
from sentence_transformers import InputExample
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde

import pandas as pd

from notebooks.utx import plot


def load_data(path: str = "data/preprocessed.feather") -> Tuple[InputExample, InputExample, InputExample]:
    ds = pd.read_feather(path)
    ds.score /= 5
    data = [InputExample(texts=[s1, s2], label=score)
            for s1, s2, score in ds.loc[:, ["sen1", "sen2", "score"]].values]
    train, test = train_test_split(data, train_size=.9, random_state=0)
    train, valid = train_test_split(train, train_size=.8, random_state=0)
    return train, valid, test


def plot_kde(ds: List[InputExample], name: str):
    scores = [t.label for t in ds]
    (
        plot(0, 1, 100)
        .function(gaussian_kde(scores))
        .labels(name, "Score", "Density")
        .show()
    )


if __name__ == '__main__':
    train, valid, test = load_data()
    plot_kde(train, "Train Dataset")
    plot_kde(valid, "Validation Dataset")
    plot_kde(test, "Test Dataset")
    print()
