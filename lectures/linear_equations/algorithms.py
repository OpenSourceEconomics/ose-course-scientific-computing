"""This module contains the algorithms for the lecture in linear equations.

The materials follow [M2004]_ (Chapter 2). The python code heavily draws on [R2020]_ and [F2019]_.

References
----------
.. [F2019] Foster, J. T. (2019). Numerical Methods and Computer Programming. e-book. URL:
           https://johnfoster.pge.utexas.edu/numerical-methods-book

.. [M2004] Miranda, M. J., & Fackler, P. L. (2004). Applied Computational Economics
           and Finance. MIT Press.

.. [R2020] Romero-Aguilar, R. (2020). CompEcon: A Python version of Miranda and Fackler's CompEcon
           toolbox. Available at https://github.com/randall-romero/CompEconR.
"""
import numpy as np
from scipy.linalg import lu

eps = np.sqrt(np.spacing(1.0))


def forward_substitution(lower_triangular, b):
    """Perform forward substitution to solve a system of linear equations.

    Solves a linear equation of type :math:`Ax = b` when for a *lower triangular* matrix
    :math:`A` of dimension :math:`n \\times n` and vector :math:`b` of length :math:`n`.
    The forward subsititution algorithm can be represented as:

    .. math::

       x_i = \\left ( b_i - \\sum_{j=1}^{i-1} a_{ij}x_j \\right )/a_{ij}


    Parameters
    ----------
    lower_triangular : numpy.ndarray
        Lower triangular matrix of dimension :math:`n \\times n`.
    b : numpy.ndarray
        Vector of length :math:`n`.

    Returns
    -------
    x : numpy.ndarray
        Solution of the linear equations. Vector of length :math:`n`.

    """
    # Get number of rows
    n = lower_triangular.shape[0]

    # Allocating space for the solution vector
    x = np.zeros_like(b, dtype=np.double)

    # Here we perform the forward-substitution.
    # Initializing  with the first row.
    x[0] = b[0] / lower_triangular[0, 0]

    # Looping over rows in reverse (from the bottom  up),
    # starting with the second to last row, because  the
    # last row solve was completed in the last step.
    for i in range(1, n):
        x[i] = (b[i] - np.dot(lower_triangular[i, :i], x[:i])) / lower_triangular[i, i]

    return x


def backward_substitution(upper_triangular, b):
    """Perform backward substitution to solve a system of linear equations.

    Solves a linear equation of type :math:`Ax = b` when for an *upper triangular* matrix
    :math:`A` of dimension :math:`n \\times n` and vector :math:`b` of length :math:`n`.
    The algorithm can be represented as:

    .. math::

       x_i = \\left ( b_i - \\sum_{j=1}^{i-1} a_{ij}x_j \\right )/a_{ij}

    The elements of :math:`x` are computed recursively.

    Parameters
    ----------
    upper_triangular : numpy.ndarray
        Lower triangular matrix of dimension :math:`n \\times n`.
    b : numpy.array
        Vector of length :math:`n`.

    Returns
    -------
    x : numpy.array
        Solution of the linear equations. Vector of length :math:`n`.

    """
    # Get number of rows.
    n = upper_triangular.shape[0]

    xcomp = np.zeros(n)

    for i in range(n - 1, -1, -1):
        tmp = b[i]
        for j in range(n - 1, i, -1):
            tmp -= xcomp[j] * upper_triangular[i, j]

        xcomp[i] = tmp / upper_triangular[i, i]

    return xcomp


def solve(a, b):
    """Solve linear equations using L-U factorization.

    Solves a linear equation of type :math:`Ax = b` when for a nonsingular square matrix
    :math:`A` of dimension :math:`n \\times n` and vector :math:`b` of length :math:`n`. Decomposes
    Algorithm decomposes matrix :math:`A` into the product of lower and upper triangular matrices.
    The linear equations can then be solved using a combination of forward and backward
    substitution.

    Two stages of the L-U algorithm:

    1. Factorization using Gaussian elimination: :math:`A=LU` where :math:`L` denotes
    a row-permuted lower triangular matrix. :math:`U` denotes a row-permuted upper
    triangular matrix.

    2. Solution using forward and backward substitution. The factored linear equation of step 1 can
    be expressed as

    .. math::

       Ax = (LU)x = L(Ux) = b

    The forward substitution algorithm solves :math:`Ly = b` for y. The backward substitution
    algorithm then solves :math:`Ux = y` for :math:`x`.

    Parameters
    ----------
    a : numpy.ndarray
        Matrix of dimension :math:`n \\times n`
    b : numpy.ndarray
        Vector of length :math:`n`.

    Returns
    -------
    x : numpy.ndarray
        Solution of the linear equations. Vector of length :math:`n`.

    Example
    -------
    >>> b = np.array([10, 8, -3])
    >>> a = np.array([[-3,2, 3], [-3,2,1], [3,0,0]])
    >>> solve(a, b)
    array([-1, 3, 1])

    """
    # Step 1: Factorization using scipy function lu.
    _, l, u = lu(a)

    # Step 2: Solution using forward and backward substitution.
    y = forward_substitution(l, b)
    x = backward_substitution(u, y)

    return x


def gauss_jacobi(a, b, x0=None, maxit=1000, tol=eps):
    """
    Solves linear equation of type $Ax = b$ using Gauss-Jacobi iterations.

    :param a n.n numpy array
    :param b n numpy array
    :param x0 n numpy array of starting values, default b
    :param maxit int, maximum number of iterations
    :param tol: float, convergence tolerance
    :return: n numpy array

    """
    conv = []

    if x0 is None:
        x = b.copy()
    else:
        x = x0

    q = np.diag(np.diag(a))
    for _ in range(maxit):
        dx = solve(q, b - a @ x)

        x += dx
        conv.append(np.linalg.norm(dx))

        if np.linalg.norm(dx) < tol:
            return x, conv

    raise StopIteration


def gauss_seidel(a, b, x0=None, lambda_=1.0, maxit=1000, tol=eps):
    """
    Solves linear equation of type $Ax = b$ using Gauss-Seidel iterations.

    :param a: n.n numpy array
    :param b: n numpy array
    :param x0: n numpy array of starting values, default b
    :param lambda_: float, over-relaxation parameter
    :param maxit: int, maximum number of iterations
    :param tol: float, convergence tolerance
    :return: n numpy array

    """
    conv = []

    if x0 is None:
        x = b.copy()
    else:
        x = x0

    q = np.tril(a)
    for _ in range(maxit):
        dx = solve(q, b - a @ x)
        x += lambda_ * dx
        conv.append(np.linalg.norm(dx))

        if np.linalg.norm(dx) < tol:
            return x, conv

    raise StopIteration
