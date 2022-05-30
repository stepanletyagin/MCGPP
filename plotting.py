import matplotlib.pyplot as plt
from config import *
from numpy import asarray
import seaborn as sns
import numpy as np


def dot_plot(data1, data2, n):
    fig, ax = plt.subplots(figsize=(30, 20))
    ax.set_title(g_movement_type, fontsize=50)
    ax.set_xlabel("knee", fontsize=40)
    ax.set_ylabel("hip", fontsize=40)
    plt.plot(data1, data2, 'bo')
    plt.grid()
    plt.show()
    fig.savefig('dot_plot' + str(n) + '.jpg')


def line_plot(data1, data2, movement_type, ax1, ax2):
    if ax2 == 'angle_knee':
        ax_name = 'KJ'
    if ax2 == 'angle_hip':
        ax_name = 'HJ'
    fig, ax = plt.subplots(figsize=(30, 20))
    ax.set_title('Приседания', fontsize=70)
    ax.patch.set_facecolor('0.99')
    ax.set_xlabel('t, c', fontsize=70)
    ax.set_ylabel(ax_name + ', градусы', fontsize=70)
    ax.tick_params(axis='both', which='major', labelsize=60)
    plt.plot(data1, data2)
    # plt.grid()
    # plt.show()
    fig.savefig('/Users/stepanletyagin/Desktop/BMSTU/Diploma_project/python_code/plotting/'
                'mc_plotting/startup_line_plots/' + movement_type + '/' + movement_type + '_' + ax2 + '_line_plot.jpg')


def hist(data1, data2):
    fig = plt.figure(num=1, figsize=(30, 20), dpi=300, facecolor='w', edgecolor='k')
    # plt.hist2d(data1, data2, bins=(70, 70), vmax=30)
    plt.hist(data1, bins=50)
    # plt.colorbar()
    plt.tick_params(axis='both', which='major', labelsize=35)
    plt.xlabel('knee', fontsize=40)
    plt.ylabel('hip', fontsize=40)
    fig.savefig('hist_plot.jpg')


def gpp_plot(exp_data, data, series_borders, predict_states_val_num, std_err, max_abs_err, mae_err, eval_parameter):
    time = data.loc[:, 'time']
    for i in range(0, len(series_borders) - 1):  # Number of series
        test_time_start = series_borders[i + 1] - int(predict_states_val_num[i])  # start idx of predicted states

        train_data = asarray(exp_data.loc[series_borders[i]:series_borders[i + 1] - 1, eval_parameter])
        predicted_data = asarray(data.loc[test_time_start - 1:series_borders[i + 1] - 1, eval_parameter])

        fig = plt.figure(num=1, figsize=(30, 20), dpi=300, facecolor='w', edgecolor='k')
        # fig = plt.tick_params(axis='both', which='major', labelsize=16)
        ax = fig.add_subplot()
        fig.suptitle(g_param1, y=1.2, fontsize=30)
        ax.set_title('series #' + str(i) + ', STD = ' + str(round(std_err[i], 2)) + ', MAX = ' +
                     str(round(max_abs_err[i], 2)) + ', MAE = ' + str(round(mae_err[i], 2)), fontsize=35)
        ax.plot(time[series_borders[i]:series_borders[i + 1]], train_data,
                'b-', markersize=5, label=u'Observation')
        ax.plot(time[test_time_start - 1:series_borders[i + 1]],
                predicted_data, 'r-', linewidth=1, label=u'Prediction')
        # plt.axvline(x=time[test_time_start - 1])
        ax.legend(loc='upper right', fontsize=35)
        plt.tick_params(axis='both', which='major', labelsize=35)
        ax.grid()
        plt.savefig('/Users/stepanletyagin/Desktop/BMSTU/6_semestr/Coursework/python_code/plots/'
                    + g_movement_type + '/' + eval_parameter + '/GPP_plot_' + str(i) + '.jpg')
        plt.cla()
        plt.clf()
        plt.close(fig)


def gpp_dot_plot(exp_data, predicted_data, movement_type):
    fig = plt.figure(num=1, figsize=(30, 20), dpi=300, facecolor='w', edgecolor='k')
    ax = fig.add_subplot()
    ax.set_title(movement_type, fontsize=30)
    ax.plot(predicted_data.loc[:, g_param1], predicted_data.loc[:, g_param2], 'r.', linewidth=1, label=u'Prediction')
    ax.plot(exp_data.loc[:, g_param1], exp_data.loc[:, g_param2], 'b.', markersize=5, label=u'Observation')
    ax.legend(loc='upper right', fontsize=30)
    plt.xlabel('knee', fontsize=35)
    plt.ylabel('hip', fontsize=35)
    plt.grid()
    fig.savefig('/Users/stepanletyagin/Desktop/BMSTU/6_semestr/Coursework/python_code/plots/'
                + movement_type + '/GPP_dot_plot.jpg')


def gpp_plot_distr(data, train_set_l, val_set_l, time, eval_parameter, i):
    fig = plt.figure(num=1, figsize=(30, 20), dpi=300, facecolor='w', edgecolor='k')
    ax = fig.add_subplot()
    fig.suptitle(g_param1, y=1.2, fontsize=30)

    ax.plot(time[0:train_set_l], data[0:train_set_l], 'b-', markersize=5, label=u'training data')
    ax.plot(time[val_set_l - 1:len(data)], data[val_set_l - 1:len(data)], 'g-', markersize=5, label=u'testing data')
    ax.legend(loc='upper right', fontsize=35)
    plt.tick_params(axis='both', which='major', labelsize=35)
    ax.grid()
    plt.savefig('/Users/stepanletyagin/Desktop/BMSTU/6_semestr/Coursework/python_code/plots/'
                + g_movement_type + '/' + eval_parameter + '/data_distribution' + '/GPP_plot_' + str(i) + '.jpg')
    plt.cla()
    plt.clf()
    plt.close(fig)


def f(data, n):
    sigma_n = 1e-10
    epsilon = np.random.normal(loc=0, scale=sigma_n, size=n)
    f = data + epsilon
    return f


def error_dist(n):
    # Error standard deviation.
    sns.set_style(style='whitegrid')
    sns.set_palette(palette='deep')
    fig = plt.figure(num=1, figsize=(15, 10), dpi=300, facecolor='w', edgecolor='k')
    ax = fig.add_subplot()
    # Errors.
    sigma_n = 1e-10
    epsilon = np.random.normal(loc=0, scale=sigma_n, size=n)
    sns.distplot(epsilon, ax=ax)
    ax.set(title='Noise Distribution')
    fig.savefig('error_distr.jpg')


def distr_hist(data1, data2):
    fig, ax = plt.subplots(figsize=(90, 50))
    ax.plot(data1, data2, 'o', color='#06470c', markersize=20)
    ax.patch.set_facecolor('0.99')
    ax.tick_params(axis='both', which='major', labelsize=180)
    ax.set_xlabel('KJ, градусы', fontsize=170)
    ax.set_ylabel('HJ, градусы', fontsize=170)
    ax.set_xlim([-3, 75])
    fig.savefig('/Users/stepanletyagin/Desktop/BMSTU/Diploma_project/python_code/'
                'plotting/mc_plotting/startup_hist_distr/' + g_movement_type + '/' + 'hist_plot.jpg')