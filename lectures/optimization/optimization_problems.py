import numpy as np


def golden_search_problem(x):
    return x * np.cos(x ** 2)


def get_parameterization(dimension, add_noise, add_illco):
    if add_noise:
        b = 1
    else:
        b = 0

    if add_illco:
        conditioning_factor = float(add_illco) * 20
        quadratic_coeff = np.array(np.exp(np.random.random(dimension) * conditioning_factor))
        quadratic_coeff = quadratic_coeff / np.max(quadratic_coeff)
        a = quadratic_coeff
    else:
        a = np.ones(dimension)

    return a, b


def _get_test_function_gradient(x, a, b):
    return np.array(
        np.multiply(a, np.array(x) - np.ones(np.array(x).size))
    ) + b * 2 * np.pi * np.array(np.sin(2 * np.pi * (np.array(x) - np.ones(np.array(x).size))))


def get_test_function(x, a, b):
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
    x, a = np.atleast_1d(x), np.atleast_1d(a)
    dimension = len(x)

    fval = 0
    fval += 0.5 * np.sum(a * (x - np.ones(dimension)) ** 2)
    fval += b * dimension
    fval -= b * np.sum(np.cos(2 * np.pi * (x - np.ones(dimension))))

    return fval
