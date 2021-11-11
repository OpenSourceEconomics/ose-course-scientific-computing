"""Example problems for approximation lab."""
import numpy as np


def problem_reciprocal_exponential(x):
    """Return reciprocal exponential function."""
    return np.exp(-x)


def problem_kinked(x):
    """Return function with kink."""
    return np.sqrt(np.abs(x))


def problem_two_dimensions(x, y):
    """Return two dimensional example function."""
    return np.cos(x) / np.exp(y)
