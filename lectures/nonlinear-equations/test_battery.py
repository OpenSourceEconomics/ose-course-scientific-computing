"""This module contains some tests for our functions.
"""
import numpy as np
from scipy.optimize import bisect as sp_bisect
from scipy.optimize import newton as sp_newton

from algorithms import bisect
from algorithms import fixpoint
from algorithms import newton_method


def test_1():
    """ This test that the bisection method is working.
    """

    def example(x):
        return x ** 3 - 2

    y = bisect(example, 1, 2)

    np.testing.assert_almost_equal(y, 1.259921)
    np.testing.assert_almost_equal(sp_bisect(example, 1, 2), y)


def test_2():
    """ This test that the fixpoint method is working.
    """

    def example(x):
        return np.sqrt(x)

    y = fixpoint(example, 2)
    np.testing.assert_almost_equal(y, 1.0, decimal=3)


def test_3():
    """ This test that Newton method is working.
    """

    def example(x):
        return x ** 3 - 2

    def dexample(x):
        return 3 * x ** 2

    x0 = np.array(1)
    y = newton_method(example, dexample, x0)

    np.testing.assert_almost_equal(y, 1.259921)
    np.testing.assert_almost_equal(sp_newton(example, x0, fprime=dexample), y)
