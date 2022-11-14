"""Solutions for exercises from nonlinear equations lab."""
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

from labs.nonlinear_equations.nonlinear_algorithms import bisect
from labs.nonlinear_equations.nonlinear_algorithms import fixpoint
from labs.nonlinear_equations.nonlinear_algorithms import newton_method
from labs.nonlinear_equations.nonlinear_problems import bisection_test_function
from labs.nonlinear_equations.nonlinear_problems import get_cournot_problem
from labs.nonlinear_equations.nonlinear_problems import newton_pathological_example


def test_exercise_1():
    """Test for exercise 1."""
    lower, upper = 1, 2
    x, xvals = bisect(bisection_test_function, lower, upper)

    # Test ensuring that we found a root.
    np.testing.assert_almost_equal(bisection_test_function(x), 0.0)

    # Plot that shows all iterates.
    fig, ax = plt.subplots()
    ax.plot(xvals)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("x")


def test_exercise_2():
    """Test for exercise 2."""

    def func(x):
        return np.sqrt(x + 0.2)

    x, xvals = fixpoint(func, 0.4, tolerance=1e-10)
    np.testing.assert_almost_equal(x, func(x))
    print(f"Finding the fix point took {len(xvals)} iterations.")


def test_exercise_3():
    """Test for exercise 3."""

    def func(x):
        return x**4 - 2

    fig, ax = plt.subplots()
    grid = np.linspace(1.1, 1.2)
    func_v = np.vectorize(func)
    ax.plot(grid, func_v(grid))

    x = 2.3
    for it in range(50):
        print(it, x)
        step = -(x**4 - 2) / (4 * x**3)
        x += step
        if abs(step) < 1e-10:
            break
    print(x, it)


def test_exercise_4():
    """Test for exercise 4."""
    for x0 in [-0.01, 0.01]:
        x = newton_method(newton_pathological_example, 0.45)
        print(f"candidate for root {x[0][0]:+5.3f}")


def test_excerise_5():
    """Test for exercise 5."""
    alpha, beta = 0.6, np.array([0.6, 0.8])
    cournot_p = partial(get_cournot_problem, alpha, beta)

    x0 = [0.8, 0.2]
    optimize.root(cournot_p, x0, method="broyden1")
