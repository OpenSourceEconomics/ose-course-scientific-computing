import numpy as np


def process_results(df, method, res, pos):
    minimum = np.ones(res["x"].shape[0])
    df.loc[pos, :] = [
        method,
        res.nfev,
        np.linalg.norm(res.x - minimum) / np.linalg.norm(minimum),
    ]
    return df
