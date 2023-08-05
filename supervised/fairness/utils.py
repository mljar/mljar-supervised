import numpy as np


def accuracy(t, y):
    return np.round(np.sum(t == y) / t.shape[0], 4)


def selection_rate(y):
    return np.round(
        np.sum((y == 1)) / y.shape[0],
        4,
    )


def true_positive_rate(t, y):
    return np.round(
        np.sum((y == 1) & (t == 1)) / np.sum((t == 1)),
        4,
    )


def false_positive_rate(t, y):
    return np.round(
        np.sum((y == 1) & (t == 0)) / np.sum((t == 0)),
        4,
    )


def true_negative_rate(t, y):
    return np.round(
        np.sum((y == 0) & (t == 0)) / np.sum((t == 0)),
        4,
    )


def false_negative_rate(t, y):
    return np.round(
        np.sum((y == 0) & (t == 1)) / np.sum((t == 1)),
        4,
    )
