"""Python translation of saveData.m."""

import numpy as np


def saveData(filename, X, labels=None, format="%6.4e", flag=True):
    if not flag:
        return

    X = np.asarray(X)
    with open(filename, "w", encoding="utf-8") as f:
        if labels is not None:
            for i in range(X.shape[1]):
                f.write(str(labels[i]) + "\t")
            f.write("\n")
        for row in X:
            for value in row:
                f.write((format % value) + "\t")
            f.write("\n")
