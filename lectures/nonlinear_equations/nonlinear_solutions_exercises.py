import matplotlib.pyplot as plt
import numpy as np
from nonlinear_algorithms import bisect
from nonlinear_algorithms import fixpoint
from nonlinear_problems import bisection_test_function


def test_exercise_1():

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
    def func(x):
        return np.sqrt(x + 0.2)

    x, xvals = fixpoint(func, 0.4, tolerance=1e-10)
    np.testing.assert_almost_equal(x, func(x))
    print(f"Finding the fix point took {len(xvals)} iterations.")


def test_exercise_3():
    def func(x):
        return x ** 4 - 2

    fig, ax = plt.subplots()
    grid = np.linspace(1.1, 1.2)
    func_v = np.vectorize(func)
    ax.plot(grid, func_v(grid))

    x = 2.3
    for it in range(50):
        print(it, x)
        step = -(x ** 4 - 2) / (4 * x ** 3)
        x += step
        if abs(step) < 1e-10:
            break
    print(x, it)
