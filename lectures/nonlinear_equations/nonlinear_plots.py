"""Plots for nonlinear equations lecture."""
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import broyden1
from scipy.optimize import fixed_point
from scipy.optimize import newton


def plot_bisection_test_function(f):
    """Plot bisect example."""
    fig, ax = plt.subplots()
    grid = np.linspace(1, 2)

    values = []
    for value in grid:
        values.append(f(value))
    ax.plot(grid, values)

    ax.axes.axhline(0, color="grey")
    ax.set_ylabel("f(x)")
    ax.set_xlabel("x")


def plot_fixpoint_example(f):
    """Plot fixpoint example."""
    fig, ax = plt.subplots()
    grid = np.linspace(0, 2)
    vf = np.vectorize(f)
    ax.plot(grid, vf(grid))
    ax.axline([0, 0], [1, 1])


def plot_newton_pathological_example(f):
    """Plot fixpoint example."""
    fig, ax = plt.subplots()
    grid = np.linspace(-2, 2)
    values = []
    for value in grid:
        values.append(f(np.array([value]))[0])
    ax.plot(grid, values)


def plot_convergence():
    """Compare convergence of Newton's method, Broyden's method, and function iteration."""

    def f(x):
        fval = np.exp(x) - 1
        # Log x values in list x_values.
        x_values.append(x)
        return fval

    def get_log_error(x):
        return np.log10(np.abs(x)).flatten()

    # Newton's Method.
    x_values = []
    _ = newton(f, 2)
    error_newton = get_log_error(x_values)

    # Broyden's Method.
    x_values = []
    _ = broyden1(f, 2)
    error_broyden = get_log_error(x_values)

    # Function iteration.
    x_values = []
    _ = fixed_point(f, 2, xtol=1e-4)
    error_funcit = get_log_error(x_values)

    # Plot results.
    plt.figure(figsize=(10, 5))
    plt.plot(error_newton, label="Newton's Method")
    plt.plot(error_broyden, label="Broyden's Method")
    plt.plot(error_funcit, label="Function Iteration")
    plt.title(r"Convergence rates for $f(x)= exp(x)-1$ with $x_0=2$")
    plt.xlabel("Iteration")
    plt.ylabel("Log10 Error")
    plt.xlim(0, 50)
    plt.ylim(-6, 2)
    plt.legend()


def plot_newtons_method():
    """Illustrates Newton's Method  for finding the root of a function $f$.

    The code for this function is taken from the Python CompEcon toolbox by
    Randall Romero-Aguilar [RA20]_ and has been slightly altered to fit the
    style of these lecture matrials.

    References
    ----------
    .. [RA20] Randall Romero-Aguilar. A Python version of Miranda and Fackler’s
    CompEcon toolbox. 2020. URL: https://github.com/randall-romero/CompEcon.
    """
    # Define function for illustration.
    def f(x):
        return x ** 5 - 3, 5 * x ** 4

    # Set axis limits and get function values.
    xmin, xmax = 1.0, 2.55
    x0, xstar = xmax - 0.05, 3 ** (1 / 5)
    x_values = np.linspace(xmin, xmax)
    y_values, _ = f(x_values)

    # Get function values an derivates for n
    # values to illustrate algorithm.
    n = 5
    x, y = np.zeros(n), np.zeros(n)
    x[0] = x0
    for k in range(n - 1):
        y[k], dlag = f(x[k])
        x[k + 1] = x[k] - y[k] / dlag

    # Set up figure.
    plt.figure(figsize=[10, 6])
    plt.title("Newton's Method", fontsize=16)
    plt.xlim(xmin, xmax)
    ax = plt.gca()
    ax.set_xticks(x[:4].tolist() + [xstar])
    ax.set_xticklabels(["$x_0$", "$x_1$", "$x_2$", "$x_3$", "$x^*$"])
    ax.set_yticks([])

    # Plot function and steps in root finding algorithm.
    plt.plot(x_values, y_values, label="Function $f(x)=x^5 - 3$")
    plt.hlines(0, xmin, xmax, colors="k")
    plt.plot(xstar, 0, "*", color="red", markersize=20, lw=0, label="root")
    for k, (xi, xinext, yi) in enumerate(zip(x, x[1:], y)):
        plt.plot([xi, xi], [1, yi], "--", color="grey")
        plt.plot([xi, xinext], [yi, 0], "r-")
        plt.plot(xi, yi, "r.", markersize=16, lw=0, label="$x_k$" if k == 0 else "")
        plt.plot(
            xinext, 0, ".", color="orange", markersize=16, label="$x_{k+1}$" if k == 0 else "",
        )
        plt.legend(fontsize=14)

def plot_secant_method():
    """Illustrates the Secant Method which replaces the derivative in Newton’s method with an estimate.
    The code for this function is taken from the Python CompEcon toolbox by
    Randall Romero-Aguilar [RA20]_ and has been slightly altered to fit the
    style of these lecture matrials.
    References
    ----------
    .. [RA20] Randall Romero-Aguilar. A Python version of Miranda and Fackler’s
    CompEcon toolbox. 2020. URL: https://github.com/randall-romero/CompEcon.
    """
    # Define function for illustration.
    
    def f(x):
        return x**5 - 3 
    
    # Set axis limits and get function values.
    xmin, xmax = 1.0, 2.55
    x0, xstar = xmax - 0.05, 3**(1/5)
    x_values = np.linspace(xmin, xmax)
    y_values= f(x_values)
    
    # Defining the function 
    # values to illustrate algorithm.
    
    n = 4
    x = np.zeros(n)
    x[:2] = x0, x0 - 0.25
    y = f(x)
    for i in range(2,n):
        x[i] = x[i - 1] - y[i - 1] * (x[i - 1] - x[i - 2]) / (y[i - 1] - y[i - 2])
        y[i] = f(x[i])
    
    #Set up figure
    plt.figure(figsize=[10, 6])
    plt.title("Secant's Method", fontsize=16)
    plt.xlim(xmin, xmax)

    ax = plt.gca()
    ax.set_xticks( x[:4].tolist() + [xstar])
    ax.set_xticklabels(['$x_0$', '$x_1$', '$x_2$','$x_3$', '$x^*$'])
    ax.set_yticks([])
    
    #Plot function
    plt.plot(x_values,y_values, label="Function $f(x)=x^5 - 3$")
    plt.hlines(0,xmin, xmax, colors='k')
    plt.plot(xstar, 0, "*", color="red", markersize=20, lw=0, label="root")
    
    for k, (xi,xinext,yi) in enumerate(zip(x,x[1:],y)):
        plt.plot([xi,xi],[0,yi],'--', color="grey")
        plt.plot([xi,xinext],[yi, 0],'r-')
        plt.plot(xi, yi, "r.", markersize=16, lw=0, label="$x_k$" if k == 0 else "")
        plt.plot(
            xinext, 0, ".", color="orange", markersize=16, label="$x_{k+1}$" if k == 0 else "",
        )
        plt.legend(fontsize=14)
        
