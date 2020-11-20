"""Auxiliary functions for optimization lecture."""
import numpy as np


def process_results(df, method, res):
    """Add results from optimizer calls to df."""
    minimum = np.ones(res["x"].shape[0])
    dist = np.linalg.norm(res["x"] - minimum) / np.linalg.norm(minimum)
    df.loc[method, :] = [res["nfev"], dist]
    return df


def get_bounds(dimension):
    """Get array of bounds -10 and 10 of length dimension."""
    bounds = np.tile(np.nan, [dimension, 2])
    bounds[:, 0], bounds[:, 1] = -10, 10
    return bounds
