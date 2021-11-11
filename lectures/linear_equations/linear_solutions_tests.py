"""Solutions to exercises from linear equations lab."""
import time
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from labs.linear_equations.linear_algorithms import backward_substitution
from labs.linear_equations.linear_algorithms import eps
from labs.linear_equations.linear_problems import get_random_problem


def test_1():
    """Solution to exercise 1."""
    for _ in range(10):
        A, b, x_true = get_random_problem()
        x_solve = backward_substitution(A, b)
        np.testing.assert_almost_equal(x_solve, x_true)


def test_2():
    """Solution to exercise 2."""

    def benchmarking_alternatives():
        def tic():
            return time.time()

        def toc(t):
            return time.time() - t

        print(
            "{:^5} {:^5}   {:^11} {:^11} \n{}".format(
                "m", "n", "np.solve(A,b)", "dot(inv(A), b)", "-" * 40
            )
        )

        for m, n in product([1, 100], [50, 500]):
            a = np.random.rand(n, n)
            b = np.random.rand(n, 1)

            tt = tic()
            [np.linalg.solve(a, b) for _ in range(m)]
            f1 = 100 * toc(tt)

            tt = tic()
            a_inv = np.linalg.inv(a)
            [np.dot(a_inv, b) for _ in range(m)]
            f2 = 100 * toc(tt)

            print(f" {m:3}   {n:3} {f1:11.2f} {f2:11.2f}")

    benchmarking_alternatives()


def test_3():
    """Solution to exercise 3."""

    def plot_ill_problem_2(cond, err, grid):
        """Plot ill problem."""
        fig, (ax1, ax2) = plt.subplots(2)
        ax1.plot(grid, cond, label="Condition")
        ax2.plot(grid, err, label="Error")
        ax1.legend()
        ax2.legend()

    def get_ill_problem_2(p):
        """Create ill problem 2."""
        # from numerical python

        a = np.array([[1, np.sqrt(p)], [1, 1 / np.sqrt(p)]])
        b = np.array([1.0, 2.0])
        x = np.array([(2 * p - 1) / (p - 1), -np.sqrt(p) / (p - 1)])

        return a, b, x

    grid = np.linspace(0.9, 1.1)
    cond, err = [], []
    for p in grid:
        A, b, x_true = get_ill_problem_2(p)
        x_solve = np.linalg.solve(A, b)

        cond.append(np.linalg.cond(A))
        err.append(np.linalg.norm(x_solve - x_true, 1))

    plot_ill_problem_2(cond, err, grid)


def gauss_jacobi(a, b, x0=None, max_iterations=1000, tolerance=eps):
    """Solves linear equation of type :math:`Ax = b` using Gauss-Jacobi iterations.

    The algorithm follows the same solution method as the Gauss-Seidel method outlined in
    :func:`gauss_seidel` with a differing definition of the splitting matrix :math:`Q`.For the
    **Gauss-Jacobi** method, the splitting matrix :math:`Q` is set equal to the diagonal matrix
    formed from the diagonal entries of matrix :math:`A`.

    Parameters
    ----------
    a : numpy.ndarray
        Matrix of dimension :math:`n \\times n`
    b : numpy.ndrray
        Vector of length :math:`n`.
    x0 : numpy.ndarray, default None
        Array of starting values. Set to :math:`b` if None.
    max_iterations : int
        Maximum number of iterations.
    tolerance : float
        Convergence tolerance.

    Returns
    --------
    x : numpy.ndarray
        Solution of the linear equations. Vector of length :math:`n`.

    Raises
    ------
    StopIteration
        If maximum number of iterations specified by `max_iterations` is reached.
    """
    if x0 is None:
        x = b.copy()
    else:
        x = x0

    q = np.diag(np.diag(a))
    for _ in range(max_iterations):
        dx = np.linalg.solve(q, b - a @ x)

        x += dx

        if np.linalg.norm(dx) < tolerance:
            return x

    raise StopIteration
