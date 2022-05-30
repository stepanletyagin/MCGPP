import numpy as np
from sklearn.neighbors import KDTree


def create_freq_sparse_matrix(centers, states_with_lags, test_sample_idx):
    states_with_lags_len = len(states_with_lags)
    freq_array = np.zeros((states_with_lags_len, 1), dtype=object)         # Массив частот переходов (по сериям)
    freq_matrix_size = len(centers)                                        # Размер каждой матрицы частот

    dist_arr = np.zeros((states_with_lags_len, 1), dtype=object)           # Массив кротчайших расстояний
    idx_arr = np.zeros((states_with_lags_len, 1), dtype=object)            # Массив номеров ближайших соседей
    row_sum_arr = np.zeros((states_with_lags_len, 1), dtype=object)        # Массив сумм по строкам для каждой матрицы

    for n in range(0, states_with_lags_len):
        temp_series = states_with_lags[n, 0].astype(np.object)               # Текущая серия
        temp_series_len = test_sample_idx[n]                                 # Размер текущей серии
        temp_series = temp_series.T
        print("--- tem_ser_num =  ---", n)
        print("--- tem_ser_len =  ---", temp_series_len)

        temp_freq_sparse_matrix = np.zeros((3, temp_series_len), dtype=int)  # Текущая матрица переходов

        temp_dist_arr = np.zeros(temp_series_len)                            # Текущие расстояния до ближайшего центра
        temp_idx_arr = np.zeros(temp_series_len)                             # Текущий индексы ближайших центров
        temp_row_sum_arr = np.zeros(freq_matrix_size, dtype=int)

        temp_tree = KDTree(centers, leaf_size=4)                             # Построение KD-tree
        for k in range(0, temp_series_len):
            temp_state = np.array([temp_series[k, :]])                       # Поиск ближайшего соседа
            dist, closest_center_ind = temp_tree.query(temp_state, k=1)
            if k == 0:
                seq_idx_cur = closest_center_ind[0, 0]
                temp_freq_sparse_matrix[0][k] = 0                            # строка
                temp_freq_sparse_matrix[1][k] = seq_idx_cur                  # столбец
                temp_row_sum_arr[0] += 1
            else:
                seq_idx_last = seq_idx_cur
                seq_idx_cur = closest_center_ind[0, 0]
                temp_freq_sparse_matrix[0][k] = seq_idx_last                 # строка
                temp_freq_sparse_matrix[1][k] = seq_idx_cur                  # столбец
                temp_row_sum_arr[seq_idx_last] += 1

            temp_idx_arr[k] = seq_idx_cur
            temp_dist_arr[k] = dist

        unique_temp_freq_sparse_matrix, unique_counts = np.unique(temp_freq_sparse_matrix, axis=1, return_counts=True)
        unique_temp_freq_sparse_matrix[2, :] = unique_counts[:]

        freq_array[n, 0] = unique_temp_freq_sparse_matrix
        idx_arr[n, 0] = temp_idx_arr
        dist_arr[n, 0] = temp_dist_arr
        row_sum_arr[n, 0] = temp_row_sum_arr

    return freq_array, dist_arr, idx_arr, row_sum_arr
