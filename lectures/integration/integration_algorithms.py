"""Algorithms for integration lab."""
import chaospy as cp
import numpy as np


def quadrature_newton_trapezoid_one(f, a, b, n):
    """Return quadrature newton trapezoid example."""
    xvals = np.linspace(a, b, n + 1)
    fvals = np.tile(np.nan, n + 1)
    h = xvals[1] - xvals[0]

    weights = np.tile(h, n + 1)
    weights[0] = weights[-1] = 0.5 * h

    for i, xval in enumerate(xvals):
        fvals[i] = f(xval)

    return np.sum(weights * fvals)


def quadrature_newton_simpson_one(f, a, b, n):
    """Return quadrature newton simpson example."""
    if n % 2 == 0:
        raise Warning("n must be an odd integer. Increasing by 1")
        n += 1

    xvals = np.linspace(a, b, n)
    fvals = np.tile(np.nan, n)

    h = xvals[1] - xvals[0]

    weights = np.tile(np.nan, n)
    weights[0::2] = 2 * h / 3
    weights[1::2] = 4 * h / 3
    weights[0] = weights[-1] = h / 3

    for i, xval in enumerate(xvals):
        fvals[i] = f(xval)

    return np.sum(weights * fvals)


def quadrature_gauss_legendre_one(f, a, b, n):
    """Return quadrature gauss legendre example."""
    xvals, weights = np.polynomial.legendre.leggauss(n)
    xval_trans = (b - a) * (xvals + 1.0) / 2.0 + a

    fvals = np.tile(np.nan, n)
    for i, xval in enumerate(xval_trans):
        fvals[i] = ((b - a) / 2.0) * f(xval)

    return np.sum(weights * fvals)


def quadrature_gauss_legendre_two(f, a=-1, b=1, n=10):
    """Return quadrature gauss legendre example."""
    n_dim = int(np.sqrt(n))

    xvals, weight_uni = np.polynomial.legendre.leggauss(n_dim)
    xvals_transformed = (b - a) * (xvals + 1.0) / 2.0 + a

    weights = np.tile(np.nan, n_dim ** 2)
    fvals = np.tile(np.nan, n_dim ** 2)

    counter = 0
    for i, x in enumerate(xvals_transformed):
        for j, y in enumerate(xvals_transformed):
            weights[counter] = weight_uni[i] * weight_uni[j]
            fvals[counter] = f([x, y])
            counter += 1

    return ((b - a) / 2) ** 2 * np.sum(weights * np.array(fvals))


def monte_carlo_naive_one(f, a=0, b=1, n=10, seed=123):
    """Return naive monte carlo example."""
    np.random.seed(seed)
    xvals = np.random.uniform(size=n)
    fvals = np.tile(np.nan, n)
    weights = np.tile(1 / n, n)

    scale = b - a
    for i, xval in enumerate(xvals):
        fvals[i] = f(a + xval * (b - a))

    return scale * np.sum(weights * fvals)


def monte_carlo_naive_two_dimensions(f, a=0, b=1, n=10, seed=128):
    """Return naive monte carlo example (two-dimensional).

    Restricted to same integration domain for both variables.
    """
    np.random.seed(seed)
    xvals = np.random.uniform(low=a, high=b, size=2 * n).reshape(n, 2)
    volume = (b - a) ** 2

    fvals = np.tile(np.nan, n)
    weights = np.tile(1 / n, n)

    for i, xval in enumerate(xvals):
        fvals[i] = f(xval)

    return volume * np.sum(weights * fvals)


def monte_carlo_quasi_two_dimensions(f, a=0, b=1, n=10, rule="random"):
    """Return Monte Carlo example (two-dimensional).

    Corresponds to naive Monthe Carlo for `rule='random'`. Restricted to same
    integration domain for both variables.
    """
    distribution = cp.J(cp.Uniform(a, b), cp.Uniform(a, b))
    samples = distribution.sample(n, rule=rule).T
    volume = (b - a) ** 2

    fvals = np.tile(np.nan, n)
    weights = np.tile(1 / n, n)

    for i, xval in enumerate(samples):
        fvals[i] = f(xval)

    return volume * np.sum(weights * fvals)
