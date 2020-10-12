"""This module contains some tests for our functions.
"""
import numpy as np
import scipy
import pytest

from algorithms import bisect


@pytest.mark.repeat(5)
def test_1():
    """ This test ensures that the forward and backward-substitutions are working.
    """

    def example(x):
        return x ** 3 - 2

    y = bisect(example, 1, 2)

    np.testing.assert_almost_equal(y, 1.259921)
    np.testing.assert_almost_equal(scipy.optimize.bisect(example, 1, 2), y)
