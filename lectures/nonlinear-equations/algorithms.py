import numpy as np
from numpy.linalg import norm


def bisect(f, a, b, tol=1.5e-8):

    s = np.sign(f(a))

    x = (a + b) / 2
    d = (b - a) / 2

    while d > tol:
        d = d / 2
        if s == np.sign(f(x)):
            x = x + d
        else:
            x = x - d

    return x


def fixpoint(f, x0, tol=10e-5):
    """ Fixed point algorithm """
    e = 1
    while e > tol:
        x = f(x0)  # fixed point equation
        e = norm(x0 - x)  # error at the current step
        x0 = x
    return x


def newton_method(f, x0, tol=1.5e-8):
    x0 = np.atleast_2d(x0)

    # https://github.com/randall-romero/CompEcon/blob/master/textbook/chapter03.py
    xn = x0.copy()

    while True:
        fxn, gxn = f(xn)
        if np.linalg.norm(fxn) < tol:
            return xn[0]
        else:
            xn = xn - np.linalg.solve(gxn, fxn)


def mcp(f, x0, a, b, tol=1.5e-8):

    n = x0.shape[0]
    x = x0.copy()

    while True:
        fval, fjac = f(x)

        # probably need to define an axis for aggregation
        fhatval = np.minimum(np.maximum(fval, a - x), b - x)
        fhatjac = -np.identity(n)

        i = (fval > a - x) & (fval < b - x)
        if np.any(i):
            fhatjac[i] = fjac[i]

        if norm(fhatval) < tol:
            return x
        else:
            x = x - np.linalg.solve(fhatjac, fhatval)
