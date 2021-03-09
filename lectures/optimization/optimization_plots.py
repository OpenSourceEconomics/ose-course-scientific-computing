"""Plots for optimization lecture."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def plot_contour(f, allvecs, legend_path):
    """Plot contour graph for function f."""
    # Create array from values with at least two dimensions.
    allvecs = np.atleast_2d(allvecs)

    X, Y, Z = _get_grid(f)

    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title("objective function")
    plt.xlabel("variable $x_1$")
    plt.ylabel("variable $x_2$")
    plt.rc("text", usetex=False)
    plt.rc("font", family="serif")
    plt.plot(1, 1, "r*", markersize=10, label="minimum")
    plt.plot(4.5, -1.5, "bx", markersize=10, label="initial guess")
    plt.plot(
        np.array(allvecs)[:, 0], np.array(allvecs)[:, 1], "go", markersize=4, label=legend_path,
    )
    plt.legend()
    return plt


def _get_grid(f):
    """Create a grid for function f."""
    # create data to visualize objective function
    n = 50  # number of discretization points along the x-axis
    m = 50  # number of discretization points along the x-axis
    a = -2.0
    b = 5.0  # extreme points in the x-axis
    c = -2
    d = 5.0  # extreme points in the y-axis

    X, Y = np.meshgrid(np.linspace(a, b, n), np.linspace(c, d, m))
    Z = np.zeros(X.shape)

    argument = np.zeros(2)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            argument[0] = X[i, j]
            argument[1] = Y[i, j]
            Z[i][j] = f(argument)

    return X, Y, Z


def plot_surf(f):
    """Plot surface graph of function f."""
    X, Y, Z = _get_grid(f)

    fig = plt.figure()
    ax = fig.gca(projection="3d")

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)

    plt.xlabel("variable $x_1$")
    plt.ylabel("variable $x_2$")
    fig.colorbar(surf)
    plt.title("objective function")


# Function with strict global maximum, weak local maximum, strict local maximum


def f(x):
    """Get example for function with local optima."""
    return np.where((2 < x) & (x < 2.39), -0.15, np.exp(-0.5 * x) * np.cos(3 * x) * np.cos(x))


def plot_optima_example():
    """Plot example for multiple local optima in a function."""
    x = np.arange(0.5, 4.5, 0.01)

    plt.figure()
    plt.plot(x, f(x))
    plt.plot(0.87, -0.36, "o", color="red", markersize=10, lw=0)
    plt.plot(
        [2.12, 2.25], [-0.15, -0.15], "_", color="green", markersize=20, mew=5, lw=0, label="root"
    )
    plt.plot(4, -0.07, "o", color="blue", markersize=10, lw=0)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("x", fontsize=18)
    plt.ylabel("f(x)", fontsize=18)


def g(x):
    """Get smooth example function."""
    return np.cos(4 * x) * np.cos(0.7 * x)


def plot_true_observed_example():
    """Plot example of function where only few points are known."""
    x = np.arange(0.0, 5.0, 0.01)
    x_selection = np.array([0.5, 0.5, 1.75, 1.95, 2.9, 3.8, 4.1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    ax1.plot(x, g(x), "black")
    ax1.plot(x_selection, g(x_selection), "o", color="blue", markersize=8, lw=0)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlabel("x", fontsize=18)
    ax1.set_ylabel("g(x)", fontsize=18)
    ax1.set_title("True function", fontsize=20)

    ax2.plot(x, g(x), alpha=0)
    ax2.plot(x_selection, g(x_selection), "o", color="blue", markersize=8, lw=0)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlabel("x", fontsize=18)
    ax2.set_ylabel("g(x)", fontsize=18)
    ax2.set_title("Observed function", fontsize=20)
