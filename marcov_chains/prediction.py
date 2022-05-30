import numpy as np
from sklearn.neighbors import KDTree
from config import g_lags
from mc_kernels import *
from numpy import linalg as LA


def calc_variance(cur_state_row, markov_chain):
    expected_value = 0
    expected_value_squared = 0
    # row_sum = 0
    cur_state_col_arr = np.zeros(len(cur_state_row))
    prob_arr = np.zeros(len(cur_state_row))
    for i in range(len(cur_state_row)):
        cur_state_col = int(markov_chain[1, cur_state_row[i]])
        cur_predict = markov_chain[2, cur_state_row[i]]

        cur_state_col_arr[i] = cur_state_col
        prob_arr[i] = cur_predict
        # expected_value += centers[:, cur_state_col] * cur_predict
        # expected_value_squared += (centers[:, cur_state_col] ** 2) * cur_predict

        expected_value += cur_state_col * cur_predict
        expected_value_squared += (cur_state_col ** 2) * cur_predict

        # row_sum += cur_predict
    # print("--- row_sum_check =  ---", row_sum)
    variance = expected_value_squared - expected_value ** 2
    # line_plot(cur_state_col_arr, prob_arr)

    return variance


def predict_with_sparse(markov_chain, centers, val_set):
    val_set_arr_len = len(val_set)  # Кол-во серий
    predicted_states_arr = np.zeros((len(markov_chain), 1), dtype=object)  # Предсказание (по сериям)
    dim_lag_space = 2 * (g_lags + 1)  # Размерность пространства (с лагами)

    for n in range(val_set_arr_len):
        temp_val_set = val_set[n, 0]  # Текущая серия
        temp_val_set_len = len(temp_val_set[0, :])  # Размер текущей серии
        temp_prediction = np.zeros((dim_lag_space, temp_val_set_len))  # Предсказание (текущей серии)
        print("--- tem_ser_num =  ---", n)
        print("--- tem_ser_size =  ---", temp_val_set_len)

        temp_markov_chain = markov_chain[n, 0]  # Текущая матрица вероятностей

        temp_tree = KDTree(centers.T, leaf_size=4)
        for k in range(temp_val_set_len):
            cur_state = np.array([temp_val_set[:, k]])
            dist, cur_state_center_idx = temp_tree.query(cur_state, k=1)
            cur_state_row = np.where(temp_markov_chain[0, :] == cur_state_center_idx)
            cur_state_row = np.asarray(cur_state_row[1])

            temp_variance = calc_variance(centers, cur_state_row, temp_markov_chain)
            centers_dist = LA.norm(centers[:, 0] - centers[:, 1])
            temp_variance = (centers_dist + 0.5 * centers_dist) ** 2

            for i in range(len(cur_state_row)):
                cur_state_col = int(temp_markov_chain[1, cur_state_row[i]])
                cur_predict = temp_markov_chain[2, cur_state_row[i]]
                temp_dist = cur_state - centers[:, cur_state_col]
                if temp_variance != 0:
                    temp_prediction[:, k] += gaussian_kernel(temp_dist, temp_variance) * \
                                             cur_predict * centers[:, cur_state_col]
                else:
                    temp_prediction[:, k] += custom_kernel(temp_dist) * cur_predict * centers[:, cur_state_col]

                # temp_prediction[:, k] += cur_predict * centers[:, cur_state_col]
                # temp_prediction[:, k] += rational_quadratic(temp_dist) * cur_predict * centers[:, cur_state_col]
                # temp_prediction[:, k] += custom_kernel(temp_dist) * cur_predict * centers[:, cur_state_col]

        predicted_states_arr[n, 0] = temp_prediction

    return predicted_states_arr
