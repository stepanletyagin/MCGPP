from scipy.io import loadmat
import pandas as pd


def import_data(f_name):
    mat = loadmat(f_name)
    data_union = mat['data_union']

    vars = pd.read_csv('variables.txt', delimiter=" ")
    wlk_types = pd.read_csv('walk_types.txt', delimiter=",")

    data_union[1][0] = wlk_types
    data_union[1][1] = vars

    return data_union
