"""This module contains some tests for our functions.
"""
import numpy as np
from scipy.optimize import bisect as sp_bisect

from algorithms import bisect
from algorithms import fixpoint


def test_1():
    """ This test that the bisection method is working.
    """

    def example(x):
        return x ** 3 - 2

    y = bisect(example, 1, 2)

    np.testing.assert_almost_equal(y, 1.259921)
    np.testing.assert_almost_equal(sp_bisect(example, 1, 2), y)


def test_2():
    """ This test that the bisection method is working.
    """

    def example(x):
        return np.sqrt(x)

    y = fixpoint(example, 2)

    np.testing.assert_almost_equal(y, 1.0, decimal=3)
