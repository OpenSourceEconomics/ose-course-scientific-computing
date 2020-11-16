import numpy as np


def process_results(df, method, res, pos):
    minimum = np.ones(res["x"].shape[0])
    df.loc[pos, :] = [
        method,
        res.nfev,
        np.linalg.norm(res.x - minimum) / np.linalg.norm(minimum),
    ]
    return df


def get_bounds(dimension):
    bounds = np.tile(np.nan, [dimension, 2])
    bounds[:, 0], bounds[:, 1] = -10, 10
    return bounds
