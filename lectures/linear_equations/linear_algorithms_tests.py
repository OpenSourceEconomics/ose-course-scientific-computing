"""This module contains some tests for our functions."""
import numpy as np
import pytest

from lectures.linear_equations.linear_algorithms import backward_substitution
from lectures.linear_equations.linear_algorithms import forward_substitution
from lectures.linear_equations.linear_algorithms import gauss_seidel
from lectures.linear_equations.linear_algorithms import solve
from lectures.linear_equations.linear_problems import get_random_problem
from lectures.linear_equations.linear_solutions_tests import gauss_jacobi


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


@pytest.mark.repeat(5)
def test_4():
    """Check guard of interface."""
    a, b, _ = get_random_problem(is_diag=False)

    with pytest.raises(AssertionError):
        backward_substitution(a, b)

    with pytest.raises(AssertionError):
        backward_substitution(a, b)
