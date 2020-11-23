import numpy as np
from integration_differentiation_algorithms import quadrature_trapezoid


def test_1():
    np.testing.assert_almost_equal(quadrature_trapezoid(lambda x: 3 * x ** 2, 0, 1, 10000), 1)


# TODO: proper testing
