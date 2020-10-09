import numpy as np

from scipy.linalg import lu

eps = np.sqrt(np.spacing(1.0))


def forward_substitution(L, b):
    #    https: // johnfoster.pge.utexas.edu / numerical - methods - book / LinearAlgebra_LU.html  #
    #    Python/NumPy-implementation-of-backward-substitution

    # Get number of rows
    n = L.shape[0]

    # Allocating space for the solution vector
    y = np.zeros_like(b, dtype=np.double)

    # Here we perform the forward-substitution.
    # Initializing  with the first row.
    y[0] = b[0] / L[0, 0]

    # Looping over rows in reverse (from the bottom  up),
    # starting with the second to last row, because  the
    # last row solve was completed in the last step.
    for i in range(1, n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

    return y


def backward_substitution(U, b):
    n = U.shape[0]

    xcomp = np.zeros(n)

    for i in range(n - 1, -1, -1):
        tmp = b[i]
        for j in range(n - 1, i, -1):
            tmp -= xcomp[j] * U[i, j]

        xcomp[i] = tmp / U[i, i]

    return xcomp


def solve(A, b):

    P, L, U = lu(A)

    z = forward_substitution(L, b)
    x = backward_substitution(U, z)

    return x


def gauss_jacobi(A, b, x0=None, maxit=1000, tol=eps):
    """

    from the COmpecon repo

    Solves AX=b using Gauss-Jacobi iterations
    :param A: n.n numpy array
    :param b: n numpy array
    :param x0: n numpy array of starting values, default b
    :param maxit: int, maximum number of iterations
    :param tol: float, convergence tolerance
    :return: n numpy array
    """
    conv = list()

    if x0 is None:
        x = b.copy()
    else:
        x = x0

    Q = np.diag(np.diag(A))  # diagonal of A matrix
    for i in range(maxit):
        dx = solve(Q, b - A @ x)

        x += dx
        conv.append(np.linalg.norm(dx))

        if np.linalg.norm(dx) < tol:
            return x, conv
    print("problem")
    return x, conv


def gauss_seidel(A, b, x0=None, lambda_=1.0, maxit=1000, tol=eps):
    """
    Solves AX=b using Gauss-Seidel iterations
    :param A: n.n numpy array
    :param b: n numpy array
    :param x0: n numpy array of starting values, default b
    :param lambda_: float, over-relaxation parameter
    :param maxit: int, maximum number of iterations
    :param tol: float, convergence tolerance
    :return: n numpy array
    """

    conv = list()

    if x0 is None:
        x = b.copy()
    else:
        x = x0

    Q = np.tril(A)  # lower triangle part of A
    for i in range(maxit):
        dx = solve(Q, b - A @ x)
        x += lambda_ * dx
        conv.append(np.linalg.norm(dx))

        if np.linalg.norm(dx) < tol:
            return x, conv

    print("problem")
    return x, conv
