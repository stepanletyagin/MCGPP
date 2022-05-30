import numpy as np


def standard_error(experiment_data, predicted_data):
    standard_err = 0
    for i in range(0, len(predicted_data)):
        standard_err = (experiment_data[i] - predicted_data[i])**2 + standard_err

    standard_err = (standard_err / len(predicted_data))**0.5

    return standard_err


def max_abs(experiment_data, predicted_data):
    max_abs = np.zeros((len(experiment_data), 1))
    for i in range(0, len(predicted_data)):
        max_abs[i] = abs(experiment_data[i] - predicted_data[i])

    max_abs = max(max_abs)

    return max_abs[0]


def mae(experiment_data, predicted_data):
    mae = 0
    for i in range(0, len(predicted_data)):
        mae = abs(experiment_data[i] - predicted_data[i]) + mae

    mae = mae / len(predicted_data)

    return mae






