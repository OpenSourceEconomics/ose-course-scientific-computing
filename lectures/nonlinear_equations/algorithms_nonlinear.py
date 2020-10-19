"""Algorithms for lecture on nonlinear equations.

The materials follow Miranda and Fackler (2004, :cite:`miranda2004applied`) (Chapter 3).
The python code draws on Romero-Aguilar (2020, :cite:`CompEcon`).
"""
from functools import partial

import numpy as np
from scipy import optimize


def bisect(f, a, b, tolerance=1.5e-8):
    """Apply bisect method to root finding problem.

    Iterative procedure to find the root of a continuous real-values function :math:`f(x)` defined
    on a bounded interval of the real line. Define interval :math:`[a, b]` that is known to contain
    or bracket the root of :math:`f` (i.e. the signs of :math:`f(a)` and :math:`f(b)` must differ).
    The given interval :math:`[a, b]` is then repeatedly bisected into subintervals of equal length.
    Each iteration, one of the two subintervals has endpoints of different signs (thus containing
    the root of :math:`f`) and is again bisected until the size of the subinterval containing the
    root reaches a specified convergence tolerance.

    Parameters
    ----------
    f : callable
        Continuous, real-valued, univariate function :math:`f(x)`.
    a : int or float
        Lower bound :math:`a` for :math:`x \\in [a,b]`.
    b : int or float
        Upper bound :math:`a` for :math:`x \\in [a,b]`. Select :math:`a` and :math:`b` so
        that :math:`f(b)` has different sign than :math:`f(a)`.
    tolerance : float
        Convergence tolerance.

    Returns
    -------
    x : float
        Solution to the root finding problem within specified tolerance.

    Examples
    --------
    >>> x = bisect(f=lambda x : x ** 3 - 2, a=1, b=2)
    >>> round(x, 4)
    1.2599

    """
    # Get sign for f(a).
    s = np.sign(f(a))

    # Get staring values for x and interval length.
    x = (a + b) / 2
    d = (b - a) / 2

    # Continue operation as long as d is above the convergence tolerance threshold.
    # Update x by adding or subtracting value of d depending on sign of f.
    while d > tolerance:
        d = d / 2
        if s == np.sign(f(x)):
            x += d
        else:
            x -= d

    return x


def fixpoint(f, x0, tolerance=10e-5):
    """Compute fixed point using function iteration.

    Parameters
    ----------
    f : callable
        Function :math:`f(x)`.
    x0 : float
        Initial guess for fixed point (starting value for function iteration).
    tolerance : float
        Convergence tolerance (tolerance < 1).

    Returns
    -------
    x : float
        Solution of function iteration.

    Examples
    --------
    >>> fixpoint(f=lambda x : x**0.5, x0=0.4, tolerance=1e-18)
    1.0

    """
    e = 1
    while e > tolerance:
        # Fixed point equation.
        x = f(x0)
        # Error at the current step.
        e = np.linalg.norm(x0 - x)
        x0 = x
    return x


def newton_method(f, x0, tolerance=1.5e-8):
    """Apply Newton's method to solving nonlinear equation.

    Solve equation using successive linearization, which replaces the nonlinear problem
    by a sequence of linear problems whose solutions converge to the solution of the nonlinear
    problem.

    Parameters
    ----------
    f : callable
        (Univariate) function :math:`f(x)`.
    x0 : float
        Initial guess for the root of :math:`f`.
    tolerance : float
        Convergence tolerance.

    Returns
    -------
    xn : float
        Solution of function iteration.

    """
    x0 = np.atleast_2d(x0)

    # https://github.com/randall-romero/CompEcon/blob/master/textbook/chapter03.py
    xn = x0.copy()

    while True:
        fxn, gxn = f(xn)
        if np.linalg.norm(fxn) < tolerance:
            return xn[0]
        else:
            xn = xn - np.linalg.solve(gxn, fxn)


def mcp_minmax(f, x0, a, b):
    """Apply minmax root finding formulation to mixed complementarity problem.

    Function utilizes Broyden's method for solution using the function
    :func:`scipy.optimize.root`.

    Parameters
    ----------
    f : callable
        Function :math:`f(x)`.
    x0 : float
        Initial guess to root finding problem.
    a : float
        Lower bound :math:`a`.
    b : float
        Upper bound :math:`b`.

    Returns
    -------
    rslt : float
    """
    # Define minmax formulation.
    def wrapper(f, a, b, x):
        fval = f(x)[0]
        return np.fmin(np.fmax(fval, a - x), b - x)

    # Apply partial function to minmax wrapper to fix all arguments but x0.
    wrapper_p = partial(wrapper, f, a, b)
    # Apply scipy function to find root using Broyden's method.
    rslt = optimize.root(wrapper_p, x0, method="broyden1", options={"maxiter": 500})

    return rslt


def fischer(u, v, sign):
    """Define Fischer's function.

    .. math::

       \\phi_{i}^{\\pm}(u, v) = u_{i} + v_{i} \\pm \\sqrt{u_{i}^{2} + v_{i}^{2}}

    Parameters
    ----------
    u : float
    v : float
    sign : float or int
        Gives sign of equation. Should be either 1 or -1.

    Returns
    -------
    callable

    """
    return u + v + sign * np.sqrt(u ** 2 + v ** 2)


def mcp_fischer(f, x0, a, b):
    """Apply Fischer's function :func:`fischer` to mixed complementarity Problem.

    Parameters
    ----------
    f : callable
        Function :math:`f(x)`.
    x0 : float
        Initial guess to root finding problem.
    a : float
        Lower bound :math:`a`.
    b : float
        Upper bound :math:`b`.

    Returns
    -------
    rslt : float

    """

    def wrapper(f, a, b, x):
        b[b == np.inf] = 1000  # fisher solution quite sensitive, maybe good exercise to run in
        # class.

        u_inner, v_inner, sign_inner = f(x)[0], a - x, +1.0
        u_outer, v_outer, sign_outer = fischer(u_inner, v_inner, sign_inner), b - x, -1.0
        return fischer(u_outer, v_outer, sign_outer)

    # Apply partial function to minmax wrapper to fix all arguments but x0.
    wrapper_p = partial(wrapper, f, a, b)
    # Apply scipy function to find root using Broyden's method.
    rslt = optimize.root(wrapper_p, x0, method="broyden1", options={"maxiter": 500})

    return rslt
