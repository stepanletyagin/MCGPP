from importfile import import_data
from initialization import assign_val
from resampling import time_resampling
from gaussian_process_regresion import gaussian_process_prediction_spl
from gaussian_data import gauss_data_spl
from mc_prediction import run_mc_prediction

import time
from plotting import *


start_time = time.time()  # time of compilation

"""
Data import
"""
data_union = import_data(file)
values = assign_val(data_union)

"""
Data resampling
"""
series = time_resampling(values)

"""
Prediction with gaussian process regression 
"""
# ###
# predicted_values, series_borders, predict_states_val_count, STD, MAX, MAE = gaussian_process_prediction_spl(series)
# gpp_plot(series, predicted_values, series_borders, predict_states_val_count,
#          STD[:, 0], MAX[:, 0], MAE[:, 0], g_param1)
# gpp_plot(series, predicted_values, series_borders, predict_states_val_count,
#          STD[:, 1], MAX[:, 1], MAE[:, 1], g_param2)
# gpp_dot_plot(series.loc[:, g_param1:g_param2], predicted_values, g_movement_type)
# ###

"""
Prediction with Marcov chains
"""

states_with_lags, predicted_states, test = run_mc_prediction(series)

print("--- %s seconds ---" % (time.time() - start_time))
