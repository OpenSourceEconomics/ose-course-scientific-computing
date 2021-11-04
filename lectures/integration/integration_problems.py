"""Problems for integration lab."""
import numpy as np


def problem_smooth(x):
    """Return smooth integration problem (exponential function)."""
    return np.exp(-x)


def problem_kinked(x):
    """Return kinked integration problem (absolute value function)."""
    return np.sqrt(np.abs(x))
