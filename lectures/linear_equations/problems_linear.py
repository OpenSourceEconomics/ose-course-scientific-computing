"""Exemplary problems for lecture on linear equations."""
import numpy as np


def get_random_problem(n=2, is_diag=True):
    """Create random problem."""
    x = np.random.uniform(size=n)
    a = np.random.normal(size=(n, n))
    if is_diag:
        a = np.diag(np.diag(a))
    b = np.matmul(a, x)

    return a, b, x


def get_ill_problem_1(n):
    """Create ill problem (1)."""
    a = np.vander(1 + np.arange(n))
    b = np.tile(np.nan, n)
    for k in range(n):
        i = k + 1
        try:
            b[k] = (i ** n - 1) / (i - 1)
        except ZeroDivisionError:
            b[k] = n
    x = np.ones(n)

    return a, b, x


def get_inverse_demand_problem():
    """Create inverse demand problem (Judd, 1998, Figure 3.2., :cite:`judd1998numerical`)."""
    a = np.array([[1.0, 1.0], [1.0, -2]])
    b = np.array([10, -2.0])
    x = np.array([6.0, 4.0])

    return a, b, x
