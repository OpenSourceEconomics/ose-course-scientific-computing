"""This module contains the algorithms for the nonlinear equations lab.

The materials follow Miranda and Fackler (2004, :cite:`miranda2004applied`) (Chapter 3).
The python code draws on Romero-Aguilar (2020, :cite:`CompEcon`).
"""
import numpy as np


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
    >>> x = bisect(f=lambda x : x ** 3 - 2, a=1, b=2)[0]
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
    xvals = [x]

    while d > tolerance:
        d = d / 2
        if s == np.sign(f(x)):
            x += d
        else:
            x -= d

        xvals.append(x)

    return x, np.array(xvals)


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
    >>> import numpy as np
    >>> x = fixpoint(f=lambda x : x**0.5, x0=0.4, tolerance=1e-10)[0]
    >>> np.allclose(x, 1)
    True

    """
    e = 1
    xvals = [x0]

    while e > tolerance:
        # Fixed point equation.
        x = f(x0)
        # Error at the current step.
        e = np.linalg.norm(x0 - x)
        x0 = x
        xvals.append(x0)
    return x, np.array(xvals)


def funcit(f, x0=2):
    """Apply function iteration using the fixpoint method."""
    f_original = f
    f = lambda z: z - f_original(z)  # noqa
    x = fixpoint(f, x0)
    f = f_original
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
    x0 = np.atleast_1d(x0)

    # This is tailored to the univariate case.
    assert x0.shape[0] == 1

    xn = x0.copy()

    while True:
        fxn, gxn = f(xn)
        if np.linalg.norm(fxn) < tolerance:
            return xn
        else:
            xn = xn - fxn / gxn


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
