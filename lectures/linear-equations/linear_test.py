"""This module contains some tests for our functions.
"""
import numpy as np
import pytest

from algorithms import backward_substitution
from algorithms import forward_substitution
from problems import get_random_problem
from algorithms import gauss_jacobi
from algorithms import gauss_seidel
from algorithms import solve


@pytest.mark.repeat(5)
def test_1():
    """ This test ensures that the forward and backward-substitutions are working.
    """
    A, b, x_true = get_random_problem(is_diag=True)

    for method in [backward_substitution, forward_substitution]:
        x_solve = method(A, b)
        np.testing.assert_almost_equal(x_solve, x_true)


@pytest.mark.repeat(5)
def test_2():
    """
    Check our solve against numpy and truth.
    """
    A, b, x_true = get_random_problem(is_diag=True)

    for method in [solve, np.linalg.solve]:
        x_solve = method(A, b)
        np.testing.assert_almost_equal(x_solve, x_true)


@pytest.mark.repeat(5)
def test_3():
    """
    Check results from gauss seidel and gauss jacobi
    """
    A, b, x_true = get_random_problem(is_diag=True)

    for method in [gauss_seidel, gauss_jacobi]:
        x_solve, _ = method(A, b)
        np.testing.assert_almost_equal(x_solve, x_true)
