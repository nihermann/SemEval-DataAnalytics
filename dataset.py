from typing import List, Tuple
from sentence_transformers import InputExample
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde

import pandas as pd

from notebooks.utx import plot


def load_data(path: str = "data/preprocessed.feather") -> Tuple[List[InputExample], List[InputExample], List[InputExample]]:
    ds = pd.read_feather(path)
    ds.score /= 5
    data = [InputExample(texts=[s1, s2], label=score)
            for s1, s2, score in ds.loc[:, ["sen1", "sen2", "score"]].values]
    train, test = train_test_split(data, train_size=.9, random_state=0)
    train, valid = train_test_split(train, train_size=.8, random_state=0)
    return train, valid, test


def plot_kde(datasets: List[List[InputExample]], names: List[str]):
    get_scores = lambda ds: [t.label for t in ds]
    pl = plot(0, 1, 100).labels("Score Density Distributions", "Score", "Density")

    for data, name in zip(datasets, names):
        pl.function(gaussian_kde(get_scores(data)), label=name)
    pl.legend().show()


if __name__ == '__main__':
    train, valid, test = load_data()
    plot_kde([train, valid, test], ["Train Dataset", "Validation Dataset", "Test Dataset"])
    print()
