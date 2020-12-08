import numpy as np


def problem_runge(x):
    return 1 / (1 + 25 * x ** 2)


def problem_reciprocal_exponential(x):
    return np.exp(-x)


def problem_kinked(x):
    return np.sqrt(np.abs(x))


def problem_two_dimensions(x, y):
    return np.cos(x) / np.exp(y)
