from config import g_lags, g_movement_type, g_param1, g_param2
from sklearn.neighbors import KDTree
import numpy as np
import math
from numpy import linalg as LA
import matplotlib.pyplot as plt
from numpy import linspace
from error import *

length_scale = 1

constant_value = 1e-15

noise_level_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]
# length_scale_list = np.linspace(0.0, 10.1, num=50)


def kernel(vector, noise_level_list):
    return constant_value * math.exp((-2 * math.sin((math.pi * LA.norm(vector)) / 1) ** 2) / 1 ** 2) \
           + noise_level_list ** 2


def hyperparam_search_plot(std, max, mae, hyper_param, eval_parameter):
    fig = plt.figure(num=1, figsize=(30, 20), dpi=300, facecolor='w', edgecolor='k')
    ax = fig.add_subplot()
    # ax.set_title(fontsize=30)

    if eval_parameter == 'angle_knee':
        std_min = np.min(std[0, :])
        plt.scatter(hyper_param[np.where(std[0, :] == std[0, :].min())[0][0]], std_min, color='orange', s=100,
                    marker='o')

        max_min = np.min(max[0, :])
        plt.scatter(hyper_param[np.where(max[0, :] == max[0, :].min())[0][0]], max_min, color='orange', s=100,
                    marker='o')

        mae_min = np.min(mae[0, :])
        plt.scatter(hyper_param[np.where(mae[0, :] == mae[0, :].min())[0][0]], mae_min, color='orange', s=100,
                    marker='o')

        ax.plot(hyper_param, std[0, :], 'r-', linewidth=2.5, markersize=5, label=u'STD')
        ax.plot(hyper_param, max[0, :], 'b-', linewidth=2.5, markersize=5, label=u'MAX')
        ax.plot(hyper_param, mae[0, :], 'g-', linewidth=2.5, markersize=5, label=u'MAE')
        ax.legend(loc='upper right', fontsize=50)
        ax.tick_params(axis='both', which='major', labelsize=70)
        plt.xlabel('Шум', fontsize=60)
        plt.ylabel('KJ, градусы', fontsize=60)
        plt.grid()

        fig.savefig('/Users/stepanletyagin/Desktop/BMSTU/Diploma_project/'
                    'python_code/markov_chains/plotting/hyperparameters_search/' + g_movement_type +
                    '/MC_noise_level_error_RW_knee.jpg')
        plt.cla()
        plt.clf()
        plt.close(fig)
    if eval_parameter == 'angle_hip':
        std_min = np.min(std[1, :])
        plt.scatter(hyper_param[np.where(std[1, :] == std[1, :].min())[0][0]], std_min, color='orange', s=100,
                    marker='o')

        max_min = np.min(max[1, :])
        plt.scatter(hyper_param[np.where(max[1, :] == max[1, :].min())[0][0]], max_min, color='orange', s=100,
                    marker='o')

        mae_min = np.min(mae[1, :])
        plt.scatter(hyper_param[np.where(mae[1, :] == mae[1, :].min())[0][0]], mae_min, color='orange', s=100,
                    marker='o')

        ax.plot(hyper_param, std[1, :], 'r-', linewidth=2.5, markersize=5, label=u'STD')
        ax.plot(hyper_param, max[1, :], 'b-', linewidth=2.5, markersize=5, label=u'MAX')
        ax.plot(hyper_param, mae[1, :], 'g-', linewidth=2.5, markersize=5, label=u'MAE')
        ax.tick_params(axis='both', which='major', labelsize=70)
        ax.legend(loc='upper right', fontsize=50)
        plt.xlabel('Шум', fontsize=60)
        plt.ylabel('HJ, градусы', fontsize=60)
        plt.grid()

        fig.savefig('/Users/stepanletyagin/Desktop/BMSTU/Diploma_project/'
                    'python_code/markov_chains/plotting/hyperparameters_search/' + g_movement_type +
                    '/MC_noise_level_error_RW_hip.jpg')
        plt.cla()
        plt.clf()
        plt.close(fig)


def best_noise_level_search(markov_chain, centers, val_set, test_set):
    std = np.empty((0, 2), dtype=float)
    max = np.empty((0, 2), dtype=float)
    MAE = np.empty((0, 2), dtype=float)
    for nl in noise_level_list:
        dim_lag_space = 2 * (g_lags + 1)  # Размерность пространства (с лагами)

        n = 0
        tmp_val_set = val_set[n, 0]  # Текущая серия
        val_set_len = len(tmp_val_set[0, :])  # Размер текущей серии
        prediction = np.zeros((dim_lag_space, val_set_len))  # Предсказание (текущей серии)
        print("--- tem_ser_num =  ---", n)
        print("--- tem_ser_size =  ---", val_set_len)

        temp_markov_chain = markov_chain[n, 0]  # Текущая матрица вероятностей

        temp_tree = KDTree(centers.T, leaf_size=4)
        for k in range(val_set_len):
            cur_state = np.array([tmp_val_set[:, k]])
            dist, cur_state_center_idx = temp_tree.query(cur_state, k=1)
            cur_state_row = np.where(temp_markov_chain[0, :] == cur_state_center_idx)
            cur_state_row = np.asarray(cur_state_row[1])

            for i in range(len(cur_state_row)):
                cur_state_col = int(temp_markov_chain[1, cur_state_row[i]])
                cur_predict = temp_markov_chain[2, cur_state_row[i]]
                temp_dist = cur_state - centers[:, cur_state_col]
                prediction[:, k] += kernel(temp_dist, nl) * cur_predict * centers[:, cur_state_col]
                # temp_prediction[:, k] += cur_predict * centers[:, cur_state_col]

        temp_exp_data = test_set[n, 0]
        std_1 = standard_error(temp_exp_data[0, :], prediction[0, :])
        max_1 = max_abs(temp_exp_data[0, :], prediction[0, :])
        mae_1 = mae(temp_exp_data[0, :], prediction[0, :])

        std_2 = standard_error(temp_exp_data[1, :], prediction[1, :])
        max_2 = max_abs(temp_exp_data[1, :], prediction[1, :])
        mae_2 = mae(temp_exp_data[1, :], prediction[1, :])

        std = np.append(std, [[std_1, std_2]], axis=0)
        max = np.append(max, [[max_1, max_2]], axis=0)
        MAE = np.append(MAE, [[mae_1, mae_2]], axis=0)

    hyperparam_search_plot(std.T, max.T, MAE.T, noise_level_list, g_param1)
    hyperparam_search_plot(std.T, max.T, MAE.T, noise_level_list, g_param2)
    print("--- STOP  ---")
