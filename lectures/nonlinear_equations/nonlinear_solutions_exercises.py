import matplotlib.pyplot as plt
from nonlinear_algorithms import bisect
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
