"""Optimization problems for optimization lab."""
import numpy as np


def get_parameterization(dimension, add_noise, add_illco):
    """Get parametrization for $a_1$ and $b$ from lab exercise."""
    if add_noise:
        b = 1
    else:
        b = 0

    if add_illco:
        a = np.exp(np.random.uniform(high=20, size=dimension))
        a = a / np.max(a)
    else:
        a = np.ones(dimension)

    return a, b


def get_test_function_gradient(x, a, b):
    """Test function gradient."""
    x, a = np.atleast_1d(x), np.atleast_1d(a)
    dimension = len(x)

    grad = np.multiply(a, x - np.ones(dimension))
    grad += b * 2 * np.pi * np.sin(2 * np.pi * (x - np.ones(dimension)))

    return grad


def get_test_function(x, a, b):
    """Test function from lab exercise."""
    x, a = np.atleast_1d(x), np.atleast_1d(a)
    dimension = len(x)

    fval = 0
    for n in range(dimension):
        fval += a[n] * (x[n] - 1) ** 2

    fval = 0.5 * fval
    fval += b * dimension

    for n in range(dimension):
        fval -= b * np.cos(2 * np.pi * (x[n] - 1))

    return fval
