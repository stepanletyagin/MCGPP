import numpy as np
from config import g_mc_test_sample


def mc_data_train_sample_split(states_with_lags_s):
    states_with_lags_len = len(states_with_lags_s)
    train_sample_array_idx = np.zeros(states_with_lags_len, dtype=int)

    for n in range(0, states_with_lags_len):
        temp_series_len = len(states_with_lags_s[n][0][:][0])
        train_sample_array_idx[n] = g_mc_test_sample * temp_series_len

    return train_sample_array_idx


def mc_data_val_test_sample_split(states_with_lags_s, train_idx):
    states_with_lags_len = len(states_with_lags_s)
    val_sample_array = np.zeros((states_with_lags_len, 1), dtype=object)
    test_sample_array = np.zeros((states_with_lags_len, 1), dtype=object)

    for n in range(0, states_with_lags_len):
        temp_series_len = len(states_with_lags_s[n][0][:][0])
        if (temp_series_len - train_idx[n]) % 2 == 0:
            train_idx[n] -= 1
        temp_series_val_test_slice = states_with_lags_s[n][0][:, train_idx[n] + 1:temp_series_len]
        val_sample_array[n, 0] = temp_series_val_test_slice[:, ::2]
        test_sample_array[n, 0] = temp_series_val_test_slice[:, 1::2]

    return val_sample_array, test_sample_array
