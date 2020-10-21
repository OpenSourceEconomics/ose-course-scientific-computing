"""This module contains some tests for our functions."""
import numpy as np
import pytest
from algorithms_linear import backward_substitution
from algorithms_linear import forward_substitution
from algorithms_linear import gauss_seidel
from algorithms_linear import solve
from exercises_solutions import gauss_jacobi
from problems_linear import get_random_problem


@pytest.mark.repeat(5)
def test_1():
    """Ensure that the forward and backward-substitutions are working."""
    a, b, x_true = get_random_problem(is_diag=True)

    for method in [backward_substitution, forward_substitution]:
        x_solve = method(a, b)
        np.testing.assert_almost_equal(x_solve, x_true)


@pytest.mark.repeat(5)
def test_2():
    """Check our solve against numpy and truth."""
    a, b, x_true = get_random_problem(is_diag=True)

    for method in [solve, np.linalg.solve]:
        x_solve = method(a, b)
        np.testing.assert_almost_equal(x_solve, x_true)


@pytest.mark.repeat(5)
def test_3():
    """Check results from gauss seidel and gauss jacobi."""
    a, b, x_true = get_random_problem(is_diag=True)

    for method in [gauss_seidel, gauss_jacobi]:
        x_solve = method(a, b)
        np.testing.assert_almost_equal(x_solve, x_true)
