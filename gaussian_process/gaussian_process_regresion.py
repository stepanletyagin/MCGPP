from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from series_splitting import series_splitting_index
from gaussian_data import gauss_data_spl
from error import standard_error, max_abs, mae
from plotting import error_dist
from config import *


def gaussian_process_prediction_spl(series):
    gp = GaussianProcessRegressor(kernel=kernel_1, n_restarts_optimizer=10)

    idx = series_splitting_index(series['time'])          # series borders
    predict_states_val_count = np.zeros((len(idx), 1))    # number of predicted "states" in each series (for plotting)
    data_with_predicted_states = np.zeros((idx[-1], 3))   # series with predicted states
    standard_err = np.empty((len(idx) - 1, 2), dtype=float)
    max_abs_err = np.empty((len(idx) - 1, 2), dtype=float)
    mae_err = np.empty((len(idx) - 1, 2), dtype=float)

    for i in range(0, len(idx) - 1):  # Number of series
        temp_series = series.iloc[idx[i]:idx[i + 1], :]
        temp_series = temp_series.reset_index(drop=True)
        temp_series_l = len(temp_series)
        train_set, val_set, test_set = gauss_data_spl(temp_series, i)  # data preparation

        # error_dist(temp_series_l)
        # gp = GaussianProcessRegressor()
        # grid_search = GridSearchCV(gp, param_grid=param_grid)
        # grid_search.fit(train_time, train_set)  # Prediction methods
        # print(grid_search.best_params_)

        gp.fit(train_set, val_set)  # prediction methods
        prediction = gp.predict(test_set)

        standard_err[i, 0] = standard_error(test_set[:, 0], prediction[:, 0])  # std
        standard_err[i, 1] = standard_error(test_set[:, 1], prediction[:, 1])

        max_abs_err[i, 0] = max_abs(test_set[:, 0], prediction[:, 0])  # max_abs
        max_abs_err[i, 1] = max_abs(test_set[:, 1], prediction[:, 1])

        mae_err[i, 0] = mae(test_set[:, 0], prediction[:, 0])  # mae
        mae_err[i, 1] = mae(test_set[:, 1], prediction[:, 1])

        # filling predicted states
        temp_series.loc[temp_series_l - len(test_set):temp_series_l - 1, g_param1] = prediction[:, 0]
        temp_series.loc[temp_series_l - len(test_set):temp_series_l - 1, g_param2] = prediction[:, 1]
        data_with_predicted_states[idx[i]:idx[i + 1], 0:2] = temp_series.loc[0:temp_series_l, g_param1:g_param2]
        data_with_predicted_states[idx[i]:idx[i + 1], 2] = temp_series.loc[0:temp_series_l, 'time']

        predict_states_val_count[i] = len(prediction)

    data_with_predicted_states_df = pd.DataFrame(data_with_predicted_states, columns=[g_param1, g_param2, 'time'])

    return data_with_predicted_states_df, idx, predict_states_val_count, standard_err, max_abs_err, mae_err
