"""Optimization problems for optimization lecture."""
import numpy as np


def get_parameterization(dimension, add_noise, add_illco):
    """Get parametrization."""
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
    x, a = np.atleast_1d(x), np.atleast_1d(a)
    dimension = len(x)

    grad = np.multiply(a, x - np.ones(dimension))
    grad += b * 2 * np.pi * np.sin(2 * np.pi * (x - np.ones(dimension)))

    return grad


def get_test_function(x, a, b):
    """Get test function."""
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


def get_test_function_fast(x, a, b):
    """Get fast test function."""
    x, a = np.atleast_1d(x), np.atleast_1d(a)
    dimension = len(x)

    fval = 0
    fval += 0.5 * np.sum(a * (x - np.ones(dimension)) ** 2)
    fval += b * dimension
    fval -= b * np.sum(np.cos(2 * np.pi * (x - np.ones(dimension))))

    return fval
