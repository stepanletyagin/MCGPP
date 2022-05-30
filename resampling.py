from config import *
from series_splitting import series_splitting_index
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd


def time_resampling(values):
    values = values.T
    idx = series_splitting_index(values['time'])  # values.iloc['time']
    series_inter = pd.DataFrame(columns=values.columns)
    temp_series_inter = pd.DataFrame(columns=values.columns)

    for i in range(0, len(idx) - 1):    # number of series
        temp_series = values.iloc[idx[i]:idx[i + 1], :]
        temp_series = temp_series.drop_duplicates()
        temp_series = temp_series.reset_index(drop=True)
        interp_time = np.arange(min(temp_series['time']), max(temp_series['time']), tau)

        columns = list(temp_series)
        for k in columns:
            f = interp1d(temp_series['time'], temp_series[k])
            temp_series_inter[k] = f(interp_time)
        # The python interp1d doesn't have query points in itself,
        # but instead creates a function which you then use to get your new data points

        # temp_series_inter['time'] = np.arange(0, len(temp_series_inter) / 10, tau)
        series_inter = pd.concat([series_inter, temp_series_inter], ignore_index=True)
        temp_series_inter = temp_series_inter.iloc[0:0]     # Clear temp DataFrame

    series_inter = series_inter.sort_values(by='time', ignore_index=True)
    series_inter['time'] = np.arange(0, len(series_inter) / 10, tau)

    return series_inter
