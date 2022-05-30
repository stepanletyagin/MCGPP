import numpy as np
from config import *
from plotting import gpp_plot_distr


def gauss_data_spl(series, j):
    series_l = len(series)                                      # length of temp series
    train_set_l = int(series_l * test_sample)                   # length of train set
    axis0_train_size = int(g_window * 2 + 2)                    # number of columns of training set
    axis1_train_size = train_set_l - g_window                   # number of rows of training set

    # Recording knee hip joint values in trainig set
    gauss_data = np.empty((axis1_train_size, axis0_train_size), dtype=float)
    train_param1 = np.asarray(series.loc[0:train_set_l, g_param1]).T
    train_param2 = np.asarray(series.loc[0:train_set_l, g_param2]).T

    time = np.asarray(series.loc[0:series_l, 'time'])

    for i in range(0, axis1_train_size):
        gauss_data[i, ::2] = train_param1[i:(i + g_window + 1)]   # even values for knee angle
        gauss_data[i, 1::2] = train_param2[i:(i + g_window + 1)]  # odd values for hip angle

    # Recording knee hip joint values in validation set
    axis0_test_size = int(g_window * 2)
    axis1_test_size = (series_l - train_set_l) - g_window         # test set size
    # axis1_test_size = int(series_l - series_l * test_sample)

    test_set = np.empty((axis1_test_size, axis0_test_size), dtype=float)
    test_param1 = series.loc[int(series_l * test_sample + 1):series_l, g_param1]
    test_param2 = series.loc[int(series_l * test_sample + 1):series_l, g_param2]

    for i in range(0, axis1_test_size):
        test_set[i, ::2] = test_param1[i:(i + g_window)]
        test_set[i, 1::2] = test_param2[i:(i + g_window)]

    train_set = gauss_data[:, 0:g_window * 2]
    val_set = gauss_data[:, g_window * 2:axis0_train_size]

    # gpp_plot_distr(series.loc[:, g_param1], len(train_set[:, 0]), len(val_set[:, 0]), time, g_param1, j)
    # gpp_plot_distr(series.loc[:, g_param2], len(train_set[:, 1]), len(val_set[:, 1]), time, g_param2, j)

    return train_set, val_set, test_set
