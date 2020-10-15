import numpy as np
from numpy.linalg import norm
from functools import partial
from scipy.optimize import root


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


def mcp_minmax(f, x0, a, b):
    def wrapper(f, a, b, x):
        fval = f(x)[0]
        return np.fmin(np.fmax(fval, a - x), b - x)

    options = dict()
    options["maxiter"] = 500

    wrapper_p = partial(wrapper, f, a, b)
    rslt = root(wrapper_p, x0, method="broyden1", options=options)

    return rslt


def fisher(u, v, sign):
    return u + v + sign * np.sqrt(u ** 2 + v ** 2)


def mcp_fisher(f, x0, a, b):
    def wrapper(f, a, b, x):
        b[b == np.inf] = 1000  # fisher solution quite sensitiv, maybe good exercise to run in
        # class.

        u_inner, v_inner, sign_inner = f(x)[0], a - x, +1.0
        u_outer, v_outer, sign_outer = fisher(u_inner, v_inner, sign_inner), b - x, -1.0
        return fisher(u_outer, v_outer, sign_outer)

    options = dict()
    options["maxiter"] = 500

    wrapper_p = partial(wrapper, f, a, b)
    rslt = root(wrapper_p, x0, method="broyden1", options=options)

    return rslt
