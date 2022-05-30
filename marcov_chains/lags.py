import numpy as np
from config import *
from series_splitting import series_splitting_index


"""
Функция создает матрицы из векторов состояний в пространстве лагов вида:

[x4 x5 x6 x7 x8...
y4 y5 y6 y7 y8...
...
x0 x1 x2 x3 x4...
y0 y1 y2 y3 y4...], при размерности лагового пространства (g_lags) 2 
                                    и величине лага (g_lag_step) 2.

На четных строках значения 
углов в коленном суставе (параметр 1), на нечетных строках значения углов в тазобедренном 
суставе (параметр 2).

На вход подаетсся Data Frame, разбитый на серии. На выходе массив матриц из векторов 
в пространстве лагов. 
"""


def create_lag_space(series):
    idx = series_splitting_index(series['time'])                            # series borders
    states_with_lag_space = np.zeros((len(idx) - 1, 1), dtype=object)  # array of states for all series
    dim_lag_space = 2 * (g_lags + 1)

    for n in range(0, len(idx) - 1):
        temp_series = series.iloc[idx[n]:idx[n + 1], :]                     # temp series for creating states
        temp_series = temp_series.reset_index(drop=True)
        temp_series_l = len(temp_series)
        lag_space_len = temp_series_l - (g_lags * g_lag_step)               # lag space length
        temp_series_with_lag_space = np.zeros((2 * (g_lags + 1), lag_space_len))

        for i in range(0, g_lags + 1):
            temp_end_idx = temp_series_l - i * g_lag_step - 1
            temp_start_idx = temp_end_idx - lag_space_len + 1
            temp_series_with_lag_space[2 * i, :] = temp_series.loc[temp_start_idx:temp_end_idx, g_param1]
            temp_series_with_lag_space[2 * i + 1, :] = temp_series.loc[temp_start_idx:temp_end_idx, g_param2]

        states_with_lag_space[n, 0] = temp_series_with_lag_space

    return states_with_lag_space, dim_lag_space
