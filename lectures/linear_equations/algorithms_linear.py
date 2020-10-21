"""This module contains the algorithms for the lecture in linear equations.

The materials follow Miranda and Fackler (2004, :cite:`miranda2004applied`) (Chapter 2).
The python code heavily draws on Romero-Aguilar (2020, :cite:`CompEcon`) and
Foster (2019, :cite:`foster2019`).

"""
import numpy as np

eps = np.sqrt(np.spacing(1.0))


def forward_substitution(a, b):
    """Perform forward substitution to solve a system of linear equations.

    Solves a linear equation of type :math:`Ax = b` when for a *lower triangular* matrix
    :math:`A` of dimension :math:`n \\times n` and vector :math:`b` of length :math:`n`.
    The forward subsititution algorithm can be represented as:

    .. math::

       x_i = \\left ( b_i - \\sum_{j=1}^{i-1} a_{ij}x_j \\right )/a_{ij}


    Parameters
    ----------
    a : numpy.ndarray
        Lower triangular matrix of dimension :math:`n \\times n`.
    b : numpy.ndarray
        Vector of length :math:`n`.

    Returns
    -------
    x : numpy.ndarray
        Solution of the linear equations. Vector of length :math:`n`.

    """
    # Get number of rows
    n = a.shape[0]

    # Allocating space for the solution vector
    x = np.zeros_like(b, dtype=np.double)

    # Here we perform the forward-substitution.
    # Initializing  with the first row.
    x[0] = b[0] / a[0, 0]

    # Looping over rows in reverse (from the bottom  up), starting with the second to last row,
    # because  the last row solve was completed in the last step.
    for i in range(1, n):
        x[i] = (b[i] - np.dot(a[i, :i], x[:i])) / a[i, i]

    return x


def backward_substitution(a, b):
    """Perform backward substitution to solve a system of linear equations.

    Solves a linear equation of type :math:`Ax = b` when for an *upper triangular* matrix
    :math:`A` of dimension :math:`n \\times n` and vector :math:`b` of length :math:`n`.

    Parameters
    ----------
    a : numpy.ndarray
        Lower triangular matrix of dimension :math:`n \\times n`.
    b : numpy.ndarray
        Vector of length :math:`n`.

    Returns
    -------
    x : numpy.ndarray
        Solution of the linear equations. Vector of length :math:`n`.

    """
    # Get number of rows.
    n = a.shape[0]

    xcomp = np.zeros(n)

    for i in range(n - 1, -1, -1):
        tmp = b[i]
        for j in range(n - 1, i, -1):
            tmp -= xcomp[j] * a[i, j]

        xcomp[i] = tmp / a[i, i]

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
    >>> b = np.array([1, 2, 3])
    >>> a = np.array([[4, 0, 0], [0, 2, 0], [0, 0, 2]])
    >>> solve(a, b)
    array([0.25, 1.  , 1.5 ])

    """
    # Step 1: Factorization using scipy function lu.
    l, u = naive_LU(a)

    # Step 2: Solution using forward and backward substitution.
    y = forward_substitution(l, b)
    x = backward_substitution(u, y)

    return x


def gauss_seidel(a, b, x0=None, lambda_=1.0, max_iterations=1000, tolerance=eps):
    """
    Solves linear equation of type :math:`Ax = b` using Gauss-Seidel iterations.

    The algorithm follows the same solution method as the Gauss-Jacobi method outlines in
    :func:`gauss_jacobi` with a differing definition of the splitting matrix :math:`Q`. For
    the Gauss-Seidel method, :math:`Q` is the upper triangular matrix formed from the upper
    triangular elements of :math:`A`.

    Parameters
    ----------
    a : numpy.ndarray
        Matrix of dimension :math:`n \\times n`
    b : numpy.ndrray
        Vector of length :math:`n`.
    x0 : numpy.ndarray, default None
        Array of starting values. Set to be if None.
    lambda_ : float
        Over-relaxation parameter which may accelerate convergence of the algorithm
        for :math:`1 < \\lambda < 2`.
    max_iterations : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
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

    q = np.tril(a)
    for _ in range(max_iterations):
        dx = np.linalg.solve(q, b - a @ x)
        x += lambda_ * dx

        if np.linalg.norm(dx) < tolerance:
            return x

    raise StopIteration


def naive_LU(a):

    N = a.shape[0]
    u = a.copy()
    L = np.eye(N)
    for j in range(N - 1):
        lam = np.eye(N)
        gamma = u[j + 1 :, j] / u[j, j]
        lam[j + 1 :, j] = -gamma
        u = lam @ u

        lam[j + 1 :, j] = gamma
        L = L @ lam

    return L, u
