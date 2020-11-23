import numpy as np


def problem_smooth(x):
    return np.exp(-x)


def problem_kinked(x):
    return np.sqrt(np.abs(x))
