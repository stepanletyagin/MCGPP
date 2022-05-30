import matplotlib.pyplot as plt
from config import g_movement_type, g_param1, g_param2
from numpy import linspace
from error import *

from series_splitting import series_splitting_index
import math


def mc_dot_plot(exp_data, predicted_data):
    for n in range(len(exp_data)):
        temp_exp_data = exp_data[n, 0]
        temp_predicted_data = predicted_data[n, 0]
        fig = plt.figure(num=1, figsize=(30, 20), dpi=300, facecolor='w', edgecolor='k')
        ax = fig.add_subplot()
        ax.set_title(g_movement_type, fontsize=30)
        ax.plot(temp_predicted_data[0, :], temp_predicted_data[1, :], 'ro', linewidth=5, label=u'Prediction')
        ax.plot(temp_exp_data[0, :], temp_exp_data[1, :], 'b.', markersize=5, label=u'Observation')
        ax.legend(loc='upper right', fontsize=30)
        plt.xlabel('knee', fontsize=35)
        plt.ylabel('hip', fontsize=35)
        plt.grid()
        fig.savefig('/Users/stepanletyagin/Desktop/BMSTU/Diploma_project/python_code/markov_chains/plotting/dot_plot/'
                    + g_movement_type + '/GPP_dot_plot' + str(n) + '.jpg')
        plt.cla()
        plt.clf()
        plt.close(fig)


def mc_line_plot(exp_data, predicted_data, eval_parameter):
    for n in range(len(exp_data)):
        temp_exp_data = exp_data[n, 0]
        temp_predicted_data = predicted_data[n, 0]
        time_end = len(temp_exp_data[0, :]) * 0.1
        time = linspace(0, time_end, len(temp_exp_data[0, :]))

        fig = plt.figure(num=1, figsize=(30, 20), dpi=300, facecolor='w', edgecolor='k')
        ax = fig.add_subplot()
        # ax.set_title(g_movement_type, fontsize=30)
        if eval_parameter == 'angle_knee':
            temp_std = standard_error(temp_exp_data[0, :], temp_predicted_data[0, :])
            temp_max = max_abs(temp_exp_data[0, :], temp_predicted_data[0, :])
            temp_mae = mae(temp_exp_data[0, :], temp_predicted_data[0, :])

            ax.plot(time, temp_predicted_data[0, :], 'r-', linewidth=2.5, label=u'Prediction')
            ax.plot(time, temp_exp_data[0, :], 'b-', linewidth=2.5, markersize=5, label=u'Observation')
            ax.legend(loc='upper right', fontsize=30)
            plt.xlabel('time', fontsize=35)
            plt.ylabel(eval_parameter, fontsize=35)
            plt.grid()
            ax.set_title('series #' + str(n) + ', STD = ' + str(round(temp_std, 2)) + ', MAX = ' +
                         str(round(temp_max, 2)) + ', MAE = ' + str(round(temp_mae, 2)), fontsize=35)
            fig.savefig('/Users/stepanletyagin/Desktop/BMSTU/Diploma_project/'
                        'python_code/plotting/mc_plotting/prediction/angle_line_plot/' + g_movement_type + '/' +
                        eval_parameter + '/MC_line_plot' + str(n) + '.jpg')
        if eval_parameter == 'angle_hip':
            temp_std = standard_error(temp_exp_data[1, :], temp_predicted_data[1, :])
            temp_max = max_abs(temp_exp_data[1, :], temp_predicted_data[1, :])
            temp_mae = mae(temp_exp_data[1, :], temp_predicted_data[1, :])

            ax.set_title('series #' + str(n) + ', STD = ' + str(round(temp_std, 2)) + ', MAX = ' +
                         str(round(temp_max, 2)) + ', MAE = ' + str(round(temp_mae, 2)), fontsize=35)
            ax.plot(time, temp_predicted_data[1, :], 'r-', linewidth=2.5, label=u'Prediction')
            ax.plot(time, temp_exp_data[1, :], 'b-', linewidth=2.5, markersize=5, label=u'Observation')
            ax.legend(loc='upper right', fontsize=30)
            plt.xlabel('time', fontsize=35)
            plt.ylabel(eval_parameter, fontsize=35)
            plt.grid()
            fig.savefig('/Users/stepanletyagin/Desktop/BMSTU/Diploma_project/'
                        'python_code/plotting/mc_plotting/prediction/angle_line_plot/' + g_movement_type + '/' +
                        eval_parameter + '/MC_line_plot' + str(n) + '.jpg')
        plt.cla()
        plt.clf()
        plt.close(fig)


def mc_line_plot_splited(full_data, exp_data, predicted_data, eval_parameter):
    n = 0

    train_data = full_data[n, 0]
    prediction_data = predicted_data[n, 0]
    val_data = exp_data[n, 0]

    time_end = len(train_data[0, :]) * 0.1
    time = linspace(0, time_end, len(train_data[0, :]))

    train_data_end = len(train_data[0, :]) - len(prediction_data[0, :])
    train_data_time_start = (train_data_end - len(prediction_data[0, :])) * 0.1

    fig = plt.figure(num=1, figsize=(30, 20))
    ax = fig.add_subplot()
    ax.set_title(g_movement_type, fontsize=30)
    if eval_parameter == 'angle_knee':
        temp_std = standard_error(val_data[0, :], prediction_data[0, :])
        temp_max = max_abs(val_data[0, :], prediction_data[0, :])
        temp_mae = mae(val_data[0, :], prediction_data[0, :])

        ax.patch.set_facecolor('0.99')
        ax.plot(time[train_data_end:len(train_data[0, :])], prediction_data[0, :], 'r-', linewidth=2.5, label=u'Prediction')
        ax.plot(time[0:train_data_end], train_data[0, 0:train_data_end], 'b-', linewidth=2.5, markersize=5, label=u'Observation')
        # ax.plot(time, train_data[0, :], 'b-', linewidth=2.5, markersize=5,
        #         label=u'Observation')
        ax.legend(loc='upper right', fontsize=30)
        ax.set_xlim([train_data_time_start, time_end])
        plt.xlabel('time', fontsize=35)
        plt.ylabel('KJ', fontsize=35)
        # plt.grid()
        ax.set_title('STD = ' + str(round(temp_std, 2)) + ', MAX = ' +
                     str(round(temp_max, 2)) + ', MAE = ' + str(round(temp_mae, 2)), fontsize=35)
        fig.savefig('/Users/stepanletyagin/Desktop/BMSTU/Diploma_project/'
                    'MCGPP/plotting/mc_plotting/prediction/angle_line_plot_splitted/' + g_movement_type + '/' +
                    eval_parameter + '/MC_line_plot' + str(n) + '.jpg')
    if eval_parameter == 'angle_hip':
        temp_std = standard_error(val_data[1, :], prediction_data[1, :])
        temp_max = max_abs(val_data[1, :], prediction_data[1, :])
        temp_mae = mae(val_data[1, :], prediction_data[1, :])

        ax.patch.set_facecolor('0.99')
        ax.plot(time[train_data_end:len(train_data[1, :])], prediction_data[1, :], 'r-', linewidth=2.5, label=u'Prediction')
        ax.plot(time[0:train_data_end], train_data[1, 0:train_data_end], 'b-', linewidth=2.5, markersize=5, label=u'Observation')
        ax.legend(loc='upper right', fontsize=30)
        ax.set_xlim([train_data_time_start, time_end])
        plt.xlabel('time', fontsize=35)
        plt.ylabel('HJ', fontsize=35)
        # plt.grid()
        ax.set_title('STD = ' + str(round(temp_std, 2)) + ', MAX = ' +
                     str(round(temp_max, 2)) + ', MAE = ' + str(round(temp_mae, 2)), fontsize=35)
        fig.savefig('/Users/stepanletyagin/Desktop/BMSTU/Diploma_project/'
                    'MCGPP/plotting/mc_plotting/prediction/angle_line_plot_splitted/' + g_movement_type + '/' +
                    eval_parameter + '/MC_line_plot' + str(n) + '.jpg')
    plt.cla()
    plt.clf()
    plt.close(fig)


def line_plot(data1, data2):
    fig, ax = plt.subplots(figsize=(30, 20))
    # ax.set_title(movement_type, fontsize=50)
    ax.set_xlabel('state', fontsize=40)
    ax.set_ylabel('probability', fontsize=40)
    plt.plot(data1, data2)
    plt.grid()
    plt.show()
    fig.savefig('_line_plot.jpg')
    # fig.savefig('/Users/stepanletyagin/Desktop/BMSTU/Diploma_project/python_code/plotting/'
    #             'mc_plotting/startup_line_plots/' + movement_type + '/' + movement_type + '_' + ax2 + '_line_plot.jpg')


def create_lag_space_plot(series, g_lags, g_lag_step):
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


# График для изображения лагов
def lag_plot(series):
    g_lag_step = 1  # Lag step
    g_lags = 2  # Number of lags
    states_with_lag_space, dimension_of_lag_space = create_lag_space_plot(series, g_lags, g_lag_step)
    print("--- create_lag_space_finish ---")

    g_lag_step = 2  # Lag step
    g_lags = 2  # Number of lags
    states_with_lag_space1, dimension_of_lag_space = create_lag_space_plot(series, g_lags, g_lag_step)
    print("--- create_lag_space_finish ---")

    g_lag_step = 2  # Lag step
    g_lags = 1  # Number of lags
    states_with_lag_space2, dimension_of_lag_space = create_lag_space_plot(series, g_lags, g_lag_step)
    print("--- create_lag_space_finish ---")

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(90, 50))
    for n in range(len(states_with_lag_space)):
        temp_data1 = states_with_lag_space[n, 0]
        time_end = len(temp_data1[0, :]) * 0.1
        time1 = linspace(0, time_end, len(temp_data1[0, :]))
        # ax = fig.add_subplot(1, 3, 1)
        ax[0, 0].patch.set_facecolor('0.99')
        ax[0, 0].plot(time1, temp_data1[0, :], 'b', linewidth=2.5, label=u'x(t-2)')
        ax[0, 0].plot(time1, temp_data1[2, :], 'y', linewidth=2.5, label=u'x(t-1)')
        ax[0, 0].plot(time1, temp_data1[4, :], 'g', linewidth=2.5, label=u'x(t)')
        ax[0, 0].set_xlabel('t, с', fontsize=80)
        ax[0, 0].set_ylabel('KJ, градусы', fontsize=80)
        ax[0, 0].legend(loc='upper right', fontsize=55)
        ax[0, 0].set_xlim([3, 8])
        ax[0, 0].tick_params(axis='both', which='major', labelsize=80)

    for n in range(len(states_with_lag_space1)):
        temp_data2 = states_with_lag_space1[n, 0]
        time_end = len(temp_data2[0, :]) * 0.1
        time2 = linspace(0, time_end, len(temp_data2[0, :]))
        # ax[1] = fig.add_subplot(1, 3, 2)
        ax[0, 1].patch.set_facecolor('0.99')
        ax[0, 1].plot(time2, temp_data2[0, :], 'b', linewidth=2.5, label=u'x(t-4)')
        ax[0, 1].plot(time2, temp_data2[2, :], 'y', linewidth=2.5, label=u'x(t-2)')
        ax[0, 1].plot(time2, temp_data2[4, :], 'g', linewidth=2.5, label=u'x(t)')
        ax[0, 1].set_xlabel('t, c', fontsize=80)
        ax[0, 1].set_ylabel('KJ, градусы', fontsize=80)
        ax[0, 1].legend(loc='upper right', fontsize=55)
        ax[0, 1].set_xlim([3, 8])
        ax[0, 1].tick_params(axis='both', which='major', labelsize=80)

    for n in range(len(states_with_lag_space2)):
        temp_data3 = states_with_lag_space2[n, 0]
        time_end = len(temp_data3[0, :]) * 0.1
        time3 = linspace(0, time_end, len(temp_data3[0, :]))
        # ax[2] = fig.add_subplot(1, 3, 3)
        ax[0, 2].patch.set_facecolor('0.99')
        ax[0, 2].plot(time3, temp_data3[0, :], 'b', linewidth=2.5, label=u'x(t-2)')
        ax[0, 2].plot(time3, temp_data3[2, :], 'y', linewidth=2.5, label=u'x(t)')
        ax[0, 2].set_xlabel('t, с', fontsize=80)
        ax[0, 2].set_ylabel('KJ, градусы', fontsize=80)
        ax[0, 2].legend(loc='upper right', fontsize=55)
        ax[0, 2].set_xlim([3, 8])
        ax[0, 2].tick_params(axis='both', which='major', labelsize=80)

    for n in range(len(states_with_lag_space)):
        temp_data1 = states_with_lag_space[n, 0]
        time_end = len(temp_data1[0, :]) * 0.1
        time1 = linspace(0, time_end, len(temp_data1[0, :]))
        # ax = fig.add_subplot(1, 3, 1)
        ax[1, 0].patch.set_facecolor('0.99')
        ax[1, 0].plot(time1, temp_data1[1, :], 'b', linewidth=2.5, label=u'x(t-2)')
        ax[1, 0].plot(time1, temp_data1[3, :], 'y', linewidth=2.5, label=u'x(t-1)')
        ax[1, 0].plot(time1, temp_data1[5, :], 'g', linewidth=2.5, label=u'x(t)')
        ax[1, 0].set_xlabel('t, с', fontsize=80)
        ax[1, 0].set_ylabel('HJ, градусы', fontsize=80)
        ax[1, 0].legend(loc='upper right', fontsize=55)
        ax[1, 0].set_xlim([3, 8])
        ax[1, 0].tick_params(axis='both', which='major', labelsize=80)

    for n in range(len(states_with_lag_space1)):
        temp_data2 = states_with_lag_space1[n, 0]
        time_end = len(temp_data2[0, :]) * 0.1
        time2 = linspace(0, time_end, len(temp_data2[0, :]))
        # ax[1] = fig.add_subplot(1, 3, 2)
        ax[1, 1].patch.set_facecolor('0.99')
        ax[1, 1].plot(time2, temp_data2[1, :], 'b', linewidth=2.5, label=u'x(t-4)')
        ax[1, 1].plot(time2, temp_data2[3, :], 'y', linewidth=2.5, label=u'x(t-2)')
        ax[1, 1].plot(time2, temp_data2[5, :], 'g', linewidth=2.5, label=u'x(t)')
        ax[1, 1].set_xlabel('t, с', fontsize=80)
        ax[1, 1].set_ylabel('HJ, градусы', fontsize=80)
        ax[1, 1].legend(loc='upper right', fontsize=55)
        ax[1, 1].set_xlim([3, 8])
        ax[1, 1].tick_params(axis='both', which='major', labelsize=80)

    for n in range(len(states_with_lag_space2)):
        temp_data3 = states_with_lag_space2[n, 0]
        time_end = len(temp_data3[0, :]) * 0.1
        time3 = linspace(0, time_end, len(temp_data3[0, :]))
        # ax[2] = fig.add_subplot(1, 3, 3)
        ax[1, 2].patch.set_facecolor('0.99')
        ax[1, 2].plot(time3, temp_data3[1, :], 'b', linewidth=2.5, label=u'x(t-2)')
        ax[1, 2].plot(time3, temp_data3[3, :], 'y', linewidth=2.5, label=u'x(t)')
        ax[1, 2].set_xlabel('t, c', fontsize=80)
        ax[1, 2].set_ylabel('HJ, градусы)', fontsize=80)
        ax[1, 2].legend(loc='upper right', fontsize=55)
        ax[1, 2].set_xlim([3, 8])
        ax[1, 2].tick_params(axis='both', which='major', labelsize=80)

    # plt.grid()
    # plt.show()
    fig.savefig('/Users/stepanletyagin/Desktop/BMSTU/Diploma_project/python_code/'
                'plotting/mc_plotting/lag_space_vis/lag_space_line_plot/lag_plot.jpg')


def line_subplot_resampling(data1, data2):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(90, 50))
    ax[0, 0].patch.set_facecolor('0.99')
    ax[0, 0].plot(data1['time'], data1['angle_knee'], 'xkcd:sky blue', linewidth=2.5, label=u'До ресемплинга')
    ax[0, 0].set_xlabel('t, c', fontsize=80)
    ax[0, 0].set_ylabel('KJ, градусы', fontsize=80)
    ax[0, 0].legend(loc='upper right', fontsize=55)
    # ax[0, 0].set_xlim([3, 8])
    ax[0, 0].tick_params(axis='both', which='major', labelsize=80)

    ax[0, 1].patch.set_facecolor('0.99')
    ax[0, 1].plot(data2['time'], data2['angle_knee'], 'r', linewidth=2.5, label=u'После ресемплинга')
    ax[0, 1].set_xlabel('t, c', fontsize=80)
    ax[0, 1].set_ylabel('KJ, градусы', fontsize=80)
    ax[0, 1].legend(loc='upper right', fontsize=55)
    # ax[0, 0].set_xlim([3, 8])
    ax[0, 1].tick_params(axis='both', which='major', labelsize=80)

    ax[1, 0].patch.set_facecolor('0.99')
    ax[1, 0].plot(data1['time'], data1['angle_knee'], 'xkcd:sky blue', linewidth=2.5, label=u'До ресемплинга')
    ax[1, 0].set_xlabel('t, с', fontsize=80)
    ax[1, 0].set_ylabel('HJ, градусы', fontsize=80)
    ax[1, 0].legend(loc='upper right', fontsize=55)
    # ax[0, 0].set_xlim([3, 8])
    ax[1, 0].tick_params(axis='both', which='major', labelsize=80)

    ax[1, 1].patch.set_facecolor('0.99')
    ax[1, 1].plot(data2['time'], data2['angle_knee'], 'r', linewidth=2.5, label=u'После ресемплинга')
    ax[1, 1].set_xlabel('t, c', fontsize=80)
    ax[1, 1].set_ylabel('HJ, градусы', fontsize=80)
    ax[1, 1].legend(loc='upper right', fontsize=55)
    # ax[0, 0].set_xlim([3, 8])
    ax[1, 1].tick_params(axis='both', which='major', labelsize=80)

    fig.savefig('/Users/stepanletyagin/Desktop/BMSTU/Diploma_project/python_code/'
                'plotting/mc_plotting/resampling_vis/resampling_plot.jpg')


def centers_4d_plot(data):
    fig = plt.figure(dpi=300, facecolor='0.99', edgecolor='k')
    ax = fig.add_subplot(111, projection='3d')

    img = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=data[:, 3], cmap='Blues')
    # img = ax.scatter(data[:, 0], data[:, 1], data[:, 3], c=data[:, 2], cmap=plt.hot())

    fig.colorbar(img)
    fig.savefig('/Users/stepanletyagin/Desktop/BMSTU/Diploma_project/python_code/'
                'plotting/mc_plotting/centers_vis/centers_plot.jpg')


def data_splitting_plot(data):
    time_len = len(data[0, :])
    time_end = time_len * 0.1
    time = linspace(0, time_end, time_len)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(120, 50))
    ax[0].patch.set_facecolor('0.99')
    ax[0].plot(time[0:int(time_len * 0.8)], data[0, 0:int(time_len * 0.8)], 'xkcd:sky blue', linewidth=2.5,
            label=u'Обучающая выборка')
    ax[0].plot(time[int(time_len * 0.8):time_len], data[0, int(time_len * 0.8):time_len], 'r', linewidth=2.5,
            label=u'Тестовая выборка')
    ax[0].set_xlabel('t, c', fontsize=80)
    ax[0].set_ylabel('KJ, градусы', fontsize=80)
    ax[0].legend(loc='upper right', fontsize=55)
    # ax[0, 0].set_xlim([3, 8])
    ax[0].tick_params(axis='both', which='major', labelsize=80)

    ax[1].patch.set_facecolor('0.99')
    ax[1].plot(time[0:int(time_len * 0.8)], data[1, 0:int(time_len * 0.8)], 'xkcd:sky blue', linewidth=2.5,
            label=u'Обучающая выборка')
    ax[1].plot(time[int(time_len * 0.8):time_len], data[1, int(time_len * 0.8):time_len], 'r', linewidth=2.5,
            label=u'Тестовая выборка')
    ax[1].set_xlabel('t, c', fontsize=80)
    ax[1].set_ylabel('HJ, градусы', fontsize=80)
    ax[1].legend(loc='upper right', fontsize=55)
    # ax[0, 0].set_xlim([3, 8])
    ax[1].tick_params(axis='both', which='major', labelsize=80)

    fig.savefig('/Users/stepanletyagin/Desktop/BMSTU/Diploma_project/python_code/plotting/'
                'mc_plotting/data_split/data_split_plot.jpg')


def mc_line_subplot_splited(full_data, exp_data, predicted_data):
    n = 0

    train_data = full_data[n, 0]
    prediction_data = predicted_data[n, 0]
    val_data = exp_data[n, 0]

    time_end = len(train_data[0, :]) * 0.1
    time = linspace(0, time_end, len(train_data[0, :]))

    train_data_end = len(train_data[0, :]) - len(prediction_data[0, :])
    train_data_time_start = (train_data_end - len(prediction_data[0, :])) * 0.1

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(120, 50))
    ax[0].patch.set_facecolor('0.99')
    ax[0].set_title(g_movement_type, fontsize=30)

    temp_std = standard_error(val_data[0, :], prediction_data[0, :])
    temp_max = max_abs(val_data[0, :], prediction_data[0, :])
    temp_mae = mae(val_data[0, :], prediction_data[0, :])

    train_predict_data = np.concatenate((train_data[:, 0:train_data_end], prediction_data), axis=1)
    ax[0].plot(time[train_data_end - 1:len(train_data[0, :])],
               train_predict_data[0, train_data_end - 1:len(train_data[0, :])],
               'r-', linewidth=2.5, label=u'Прогноз')
    ax[0].plot(time[0:train_data_end], train_predict_data[0, 0:train_data_end], 'b-', linewidth=2.5, markersize=5,
               label=u'Экспериментальные данные')

    # ax[0].plot(time[train_data_end:len(train_data[0, :])], prediction_data[0, :], 'r-', linewidth=2.5,
    #            label=u'Prediction')
    # ax[0].plot(time[0:train_data_end], train_data[0, 0:train_data_end], 'b-', linewidth=2.5, markersize=5,
    #         label=u'Observation')
    ax[0].legend(loc='upper right', fontsize=80)
    ax[0].set_xlim([train_data_time_start, time_end])
    ax[0].set_xlabel('t, с', fontsize=90)
    ax[0].set_ylabel('KJ, градусы', fontsize=90)
    ax[0].tick_params(axis='both', which='major', labelsize=90)
    # plt.grid()
    ax[0].set_title('STD = ' + str(round(temp_std, 2)) + ', MAX = ' +
                 str(round(temp_max, 2)) + ', MAE = ' + str(round(temp_mae, 2)), fontsize=90)

    temp_std = standard_error(val_data[1, :], prediction_data[1, :])
    temp_max = max_abs(val_data[1, :], prediction_data[1, :])
    temp_mae = mae(val_data[1, :], prediction_data[1, :])

    ax[1].patch.set_facecolor('0.99')
    # ax[1].plot(time[train_data_end:len(train_data[1, :])], prediction_data[1, :], 'r-',
    #            linewidth=2.5, label=u'Prediction')
    # ax[1].plot(time[0:train_data_end], train_data[1, 0:train_data_end], 'b-',
    #            linewidth=2.5, label=u'Observation')
    ax[1].plot(time[train_data_end - 1:len(train_data[1, :])],
               train_predict_data[1, train_data_end - 1:len(train_data[1, :])],
               'r-', linewidth=2.5, label=u'Прогноз')
    ax[1].plot(time[0:train_data_end], train_predict_data[1, 0:train_data_end], 'b-', linewidth=2.5, markersize=5,
               label=u'Экспериментальные данные')
    ax[1].legend(loc='upper right', fontsize=80)
    ax[1].set_xlim([train_data_time_start, time_end])
    ax[1].set_xlabel('t, с', fontsize=90)
    ax[1].set_ylabel('HJ, градусы', fontsize=90)
    ax[1].tick_params(axis='both', which='major', labelsize=90)
    # plt.grid()
    ax[1].set_title('STD = ' + str(round(temp_std, 2)) + ', MAX = ' +
                 str(round(temp_max, 2)) + ', MAE = ' + str(round(temp_mae, 2)), fontsize=90)
    fig.savefig('/Users/stepanletyagin/Desktop/BMSTU/Diploma_project/'
                'MCGPP/plotting/mc_plotting/prediction/angle_line_plot_splitted/' + g_movement_type +
                '/MC_line_plot' + str(n) + '.jpg')

    plt.cla()
    plt.clf()
    plt.close(fig)


def f_g(x, variance):
    exp_val = math.sqrt(variance)
    return 1 / (exp_val * math.sqrt(2 * math.pi)) * math.exp(-1 * x ** 2 / (2 * variance))


def gaussian_kernel_plot(centers_dist):
    start_variance = (centers_dist - 0.1 * centers_dist) ** 2
    end_variance = (centers_dist + 0.1 * centers_dist) ** 2
    variance = np.linspace(start_variance, end_variance, 3)

    x = np.linspace(-20, 20, 1000, axis=0)
    # f = 1 / (exp_val * math.sqrt(2 * math.pi)) * math.exp(-1 * x ** 2 / (2 * variance))

    fig, ax = plt.subplots(figsize=(30, 20))
    ax.patch.set_facecolor('0.99')
    # ax.set_xlabel(ax1, fontsize=40)
    # ax.set_ylabel(ax_name, fontsize=40)
    ax.tick_params(axis='both', which='major', labelsize=35)
    f = np.vectorize(f_g)
    ax.plot(x, f(x, variance[0]), label=u'sigma='+str(round(variance[0], 2)))
    ax.plot(x, f(x, variance[1]), label=u'sigma='+str(round(variance[1], 2)))
    ax.plot(x, f(x, variance[2]), label=u'sigma='+str(round(variance[2], 2)))
    ax.legend(loc='upper right', fontsize=35)
    ax.set_xlim([x[0], x[len(x) - 1]])
    # plt.grid()
    # plt.show()
    fig.savefig('/Users/stepanletyagin/Desktop/BMSTU/Diploma_project/python_code/plotting/'
                'mc_plotting/prediction/sigma_plot/sigma_plot.jpg')
