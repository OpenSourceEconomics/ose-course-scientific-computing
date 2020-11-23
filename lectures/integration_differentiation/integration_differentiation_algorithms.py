import numpy as np


def quadrature_trapezoid(f, a, b, n):

    xvals = np.linspace(a, b, n + 1)
    fvals = np.tile(np.nan, n + 1)
    h = xvals[1] - xvals[0]

    weights = np.tile(h, n + 1)
    weights[0] = weights[-1] = 0.5 * h

    for i, xval in enumerate(xvals):
        fvals[i] = f(xval)

    return np.sum(weights * fvals)


def quadrature_simpson(f, a, b, n):

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


def quadrature_gauss_legendre(f, a, b, n):
    xvals, weights = np.polynomial.legendre.leggauss(n)

    fvals = np.tile(np.nan, n)
    for i, xval in enumerate(xvals):
        xval_trans = (b - a) * (xval + 1.0) / 2.0 + a
        fvals[i] = ((b - a) / 2.0) * f(xval_trans)

    return np.sum(weights * fvals)


def simps(f, a, b, N=50):
    # https: // www.math.ubc.ca / ~pwalls / math - python / integration / simpsons - rule /
    """Approximate the integral of f(x) from a to b by Simpson's rule.

    Parameters
    ----------
    f : function
        Vectorized function of a single variable
    a , b : numbers
        Interval of integration [a,b]
    N : (even) integer
        Number of subintervals of [a,b]

    Returns
    -------
    float
        Approximation of the integral of f(x) from a to b using
        Simpson's rule with N subintervals of equal length.

    Examples
    --------
    >>> simps(lambda x : 3*x**2,0,1,10)
    1.0
    """
    if N % 2 == 1:
        raise ValueError("N must be an even integer.")
    dx = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    S = dx / 3 * np.sum(y[0:-1:2] + 4 * y[1::2] + y[2::2])
    return S
