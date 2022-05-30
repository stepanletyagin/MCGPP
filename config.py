from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, RationalQuadratic as RQ, WhiteKernel, \
    ExpSineSquared as Exp, DotProduct

file = 'data.mat'
g_movement_type = 'Squat'

g_param1 = 'angle_knee'
g_param2 = 'angle_hip'

g_x_points = 4
g_y_points = 5
x_min = -13
x_max = 90
y_min = -90
y_max = -5

eps = 0.000001
tau = 1 / 10    # Interpolation step
g_lag_step = 1  # Lag step
g_lags = 2      # Number of lags
g_mc_test_sample = 0.8

test_sample = 0.8
g_window = 2

# Gaussian process set
kernel_1 = C(constant_value=1e-10, constant_value_bounds=(1e-10, 10)) * RQ(length_scale=500,
                                                                           length_scale_bounds=(1e-4, 10), alpha=50.0,
                                                                           alpha_bounds=(1e-4, 10)) + WhiteKernel(
                                                                           noise_level=0.3 ** 2)
kernel_2 = C(constant_value=1e-10) * Exp(length_scale=1, periodicity=1) + WhiteKernel(noise_level=0.3 ** 2)
kernel_3 = C(constant_value=1e-10) * RBF(length_scale=1) + WhiteKernel(noise_level=0.3 ** 2)
kernel_4 = C(constant_value=1e-10) * RBF(length_scale=1) + WhiteKernel(noise_level=0.3 ** 2)
kernel_5 = C(constant_value=1e-10) * RBF(length_scale=1) + WhiteKernel(noise_level=0.3 ** 2)

kernel_7 = C(constant_value=1e-10) * RBF(length_scale=1) + WhiteKernel(noise_level=0.3 ** 2)
kernel_8 = C(constant_value=1e-10) + WhiteKernel(noise_level=0.3 ** 2)

kernel_list = [kernel_1, kernel_2, kernel_3, kernel_4, kernel_5, kernel_7]

param_grid = {"kernel": kernel_list,
              "alpha": [0, 1e0],
              "optimizer": ["fmin_l_bfgs_b"],
              "n_restarts_optimizer": [1],
              "normalize_y": [True],
              "copy_X_train": [True],
              "random_state": [0]}
