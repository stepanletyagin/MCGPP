import numpy as np


def series_splitting_index(time):
    n = len(time)
    delta = 1
    index = np.array([])

    index = np.append(index, 0)  # First series

    for i in range(1, n):
        if abs(time[i] - time[i - 1]) > delta:
            index = np.append(index, i)

    index = np.append(index, n)  # Last series
    index = index.astype(int)
    return index
