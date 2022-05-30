import numpy as np


def row_splitting_index(row):
    n = len(row)
    delta = 1
    index = np.array([])

    index = np.append(index, 0)  # First series

    for i in range(1, n):
        if abs(row[i] - row[i - 1]) >= delta:
            index = np.append(index, i)

    index = np.append(index, n)  # Last series
    index = index.astype(int)
    return index


def calc_transition_probability_sparse_matrix(freq_array):
    prob_matrix_arr = np.zeros((len(freq_array), 1), dtype=object)

    for n in range(0, len(freq_array)):
        temp_freq_matrix = freq_array[n, 0]
        prob_matrix_cols = len(temp_freq_matrix[1, :])
        temp_prob_matrix = np.zeros((3, prob_matrix_cols), dtype=np.float64)

        temp_prob_matrix[0:2, :] = temp_freq_matrix[0:2, :]

        idx = row_splitting_index(temp_prob_matrix[0, :])                       # Разделитель строк
        for i in range(len(idx) - 1):
            temp_prob_matrix[2, idx[i]:idx[i + 1]] = temp_freq_matrix[2, idx[i]:idx[i + 1]] \
                                                     / np.sum(temp_freq_matrix[2, idx[i]:idx[i + 1]])

            # row_sum = (np.sum(temp_prob_matrix[2, idx[i]:idx[i + 1]]))
            # print("--- row_sum_check =  ---", row_sum)

        prob_matrix_arr[n, 0] = temp_prob_matrix

    return prob_matrix_arr
