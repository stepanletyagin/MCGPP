import math
from numpy import linalg as LA

alpha = 50.0

length_scale = 1
noise_level = 0.99

constant_value = 1e-15


def exp_sin_squared(vector):
    return math.exp(-2 * math.sin((math.pi * LA.norm(vector))) ** 2) + noise_level ** 2


def exp_squared(vector):
    return math.exp(-LA.norm(vector) ** 2) + noise_level ** 2


def rational_quadratic(vector):
    return constant_value * (1 + LA.norm(vector) ** 2 / (2 * alpha * length_scale)) ** (-1 * alpha) + noise_level ** 2


def custom_kernel(vector):
    return constant_value * math.exp((-2 * math.sin((math.pi * LA.norm(vector)) / 1) ** 2) / length_scale ** 2) \
           + noise_level ** 2


def gaussian_kernel(vector, variance):
    exp_val = math.sqrt(variance)
    return 1 / (exp_val * math.sqrt(2 * math.pi)) * math.exp(-1 * (LA.norm(vector)) ** 2 / (2 * variance)) \
           + noise_level ** 2
