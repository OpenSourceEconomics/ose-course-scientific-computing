"""Tests for nonlinear equations lab."""
import numpy as np
from scipy.optimize import bisect as sp_bisect

from labs.nonlinear_equations.nonlinear_algorithms import bisect
from labs.nonlinear_equations.nonlinear_algorithms import fixpoint
from labs.nonlinear_equations.nonlinear_algorithms import newton_method


def test_1():
    """Bisection method is working."""

    def example(x):
        return x**3 - 2

    y = bisect(example, 1, 2)[0]

    np.testing.assert_almost_equal(y, 1.259921)
    np.testing.assert_almost_equal(sp_bisect(example, 1, 2), y)


def test_2():
    """Fixpoint method is working."""

    def example(x):
        return np.sqrt(x)

    y = fixpoint(example, 2)[0]
    np.testing.assert_almost_equal(y, 1.0, decimal=3)


def test_3():
    """Newton method is working."""

    def _jacobian(x):
        return 3 * x**2

    def _value(x):
        return x**3 - 2

    def f(x):
        return _value(x), _jacobian(x)

    x = newton_method(f, 0.4)
    np.testing.assert_almost_equal(f(x)[0], 0)
