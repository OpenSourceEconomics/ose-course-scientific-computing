import numpy as np


def get_uniform_nodes(n, a=-1, b=1):
    return np.linspace(a, b, num=n)


def get_chebyshev_nodes(n, a=-1, b=1):

    nodes = np.tile(np.nan, n)

    for i in range(1, n + 1):
        nodes[i - 1] = (a + b) / 2 + ((b - a) / 2) * np.cos(((n - i + 0.5) / n) * np.pi)

    return nodes


def compute_interpolation_error(error):
    return np.log10(np.linalg.norm(error, np.inf))


def compute_interpolation_error_df(df):
    return np.log10(np.linalg.norm(df["Error"], np.inf))
