"""Auxiliary functions for approximation lab."""
import numpy as np


def spline_basis(x, degree, h, a):
    """Return spline."""
    nu = a + (degree - 1) * h

    if np.abs(x - nu) <= h:
        return 1 - (np.abs(x - nu)) / h
    else:
        return 0.0

    return nu


def get_uniform_nodes(n, a=-1, b=1):
    """Return uniform nodes."""
    return np.linspace(a, b, num=n)


def get_chebyshev_nodes(n, a=-1, b=1):
    """Return Chebyshev nodes."""
    nodes = np.tile(np.nan, n)

    for i in range(1, n + 1):
        nodes[i - 1] = (a + b) / 2 + ((b - a) / 2) * np.cos(((n - i + 0.5) / n) * np.pi)

    return nodes


def compute_interpolation_error(error):
    """Compute interpolation error."""
    return np.log10(np.linalg.norm(error, np.inf))


def compute_interpolation_error_df(df):
    """Compute interpolation error from a pandas.DataFrame."""
    return np.log10(np.linalg.norm(df["Error"], np.inf))
