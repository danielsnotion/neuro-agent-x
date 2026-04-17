import numpy as np


def score(pred, target):
    """
    Returns similarity score between 0 and 1
    """

    if pred.shape != target.shape:
        return 0.0

    return np.mean(pred == target)