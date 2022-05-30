from transition_frequency import create_freq_sparse_matrix
from hyper_states import hypercube_center_search
from lags import create_lag_space
from markov_chain import calc_transition_probability_sparse_matrix
from prediction import predict_with_sparse
from mc_data import mc_data_train_sample_split, mc_data_val_test_sample_split

from mc_plotting import mc_line_subplot_splited


def run_mc_prediction(series):
    states_with_lag_space, dimension_of_lag_space = create_lag_space(series)

    hypercube_centers = hypercube_center_search(dimension_of_lag_space)

    test_sample_idx = mc_data_train_sample_split(states_with_lag_space)
    val_sample, test_sample = mc_data_val_test_sample_split(states_with_lag_space, test_sample_idx)

    freq_sparse_matrix, distances_sp, indices_sp, row_sum_arr_sp \
        = create_freq_sparse_matrix(hypercube_centers, states_with_lag_space, test_sample_idx)

    prob_matrix_new = calc_transition_probability_sparse_matrix(freq_sparse_matrix)

    prediction = predict_with_sparse(prob_matrix_new, hypercube_centers.T, val_sample)

    mc_line_subplot_splited(states_with_lag_space, test_sample, prediction)

    return states_with_lag_space, prediction, test_sample
