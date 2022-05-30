import pandas as pd
from config import g_movement_type


def assign_val(data):
    mvt_idx = data[1][0].index[data[1][0]['walk_type'].apply(lambda x: g_movement_type in x)]
    values = pd.DataFrame(data[0][mvt_idx[0]])
    values = values.set_index(data[1][1].loc[:, 'variable'])

    values = values.loc[:, (values != 0).any(axis=0)]  # Deleting null columns
    values = values.T.reset_index(drop=True).T

    return values
