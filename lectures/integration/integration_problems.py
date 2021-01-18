"""Problems for integration lecture."""
import numpy as np


def problem_smooth(x):
    """Return smooth integration problem (exponential function)."""
    return np.exp(-x)


def problem_kinked(x):
    """Return kinked integration problem (absolute value function)."""
    return np.sqrt(np.abs(x))


def problem_genz_discontinuous(x, u=None, a=None):
    """Return discontinuous problem (Genz function)."""
    if u is None:
        u = np.array([0.5, 0.5])
    if a is None:
        a = np.array([5, 5])
    if x[0] > u[0] or x[1] > u[1]:
        return 0
    else:
        return np.exp((a * x).sum())
