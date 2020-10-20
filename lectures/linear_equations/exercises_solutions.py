import matplotlib.pyplot as plt

import time

import numpy as np
from problems_linear import get_random_problem
from algorithms_linear import backward_substitution, eps


for _ in range(10):
    A, b, x_true = get_random_problem()
    x_solve = backward_substitution(A, b)
    np.testing.assert_almost_equal(x_solve, x_true)


def benchmarking_alterantives():
    def tic():
        return time.time()

    def toc(t):
        return time.time() - t

    print(
        "{:^5} {:^5}   {:^11} {:^11} \n{}".format(
            "m", "n", "np.solve(A,b)", "dot(inv(A), b)", "-" * 40
        )
    )

    for m in [1, 100]:
        for n in [50, 500]:
            A = np.random.rand(n, n)
            b = np.random.rand(n, 1)

            tt = tic()
            for j in range(m):
                np.linalg.solve(A, b)

            f1 = 100 * toc(tt)

            tt = tic()
            Ainv = np.linalg.inv(A)
            for j in range(m):
                np.dot(Ainv, b)

            f2 = 100 * toc(tt)
            print(" {:3}   {:3} {:11.2f} {:11.2f}".format(m, n, f1, f2))


benchmarking_alterantives()


def get_ill_problem_2(p):
    """Create ill problem (2)."""
    # from numerical python

    a = np.array([[1, np.sqrt(p)], [1, 1 / np.sqrt(p)]])
    b = np.array([1.0, 2.0])
    x = np.array([(2 * p - 1) / (p - 1), -np.sqrt(p) / (p - 1)])

    return a, b, x


grid = np.linspace(0.9, 1.1)
cond, err = list(), list()
for p in grid:
    A, b, x_true = get_ill_problem_2(p)
    x_solve = np.linalg.solve(A, b)

    cond.append(np.linalg.cond(A))
    err.append(np.linalg.norm(x_solve - x_true, 1))


def plot_ill_problem_2(cond, err, grid):
    """Plot ill problem."""
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(grid, cond, label="Condition")
    ax2.plot(grid, err, label="Error")
    ax1.legend()
    ax2.legend()


def gauss_jacobi(a, b, x0=None, max_iterations=1000, tolerance=eps):
    """
    Solves linear equation of type :math:`Ax = b` using Gauss-Jacobi iterations.

    In the linear equation, :math:`A` denotes a matrix of dimension
    :math:`n \\times n` and :math:`b` denotes a vector of length :math:`n` The solution
    method performs especially well for larger linear equations if matrix :math`A`is
    sparse. The method achieves fairly precise approximations to the solution but
    generally does not produce *exact* solutions.

    Following the notation in Miranda and Fackler (2004, :cite:`miranda2004applied`), the linear
    equations problem can be written as

    .. math::

       Qx = b + (Q -A)x \\Rightarrow x = Q^{-1}b + (I - Q^{-1}A)x

    which suggest the iteration rule

    .. math::

       x^{(k+1)} \\leftarrow Q^{-1}b + (I - Q^{-1}A)x^{(k)}

    which, if convergent, must converge to a solution of the linear equation. For the
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
    tol : float
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
