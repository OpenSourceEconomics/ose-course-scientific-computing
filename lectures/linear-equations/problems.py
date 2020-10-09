import numpy as np


def get_random_problem(n=2, is_diag=True):

    x = np.random.uniform(size=n)

    A = np.random.normal(size=(n, n))

    if is_diag:
        A = np.diag(np.diag(A))

    b = np.matmul(A, x)

    return A, b, x


def get_ill_problem_1(n):

    v = np.vander(1 + np.arange(n))
    b = np.tile(np.nan, n)

    for k in range(n):
        i = k + 1
        try:
            b[k] = (i ** n - 1) / (i - 1)
        except ZeroDivisionError:
            b[k] = n

    x = np.ones(n)

    return v, b, x


def get_ill_problem_2(p):

    # from numerical python

    A = np.array([[1, np.sqrt(p)], [1, 1 / np.sqrt(p)]])
    b = np.array([1.0, 2.0])

    x = np.array([(2 * p - 1) / (p - 1), -np.sqrt(p) / (p - 1)])

    return A, b, x


def get_inverse_demand_problem():

    # This is from Judd Figure 3.2
    A = np.array([[1.0, 1.0], [1.0, -2]])
    b = np.array([10, -2.0])

    x = np.array([6.0, 4.0])

    return A, b, x
