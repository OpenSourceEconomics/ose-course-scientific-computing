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


def f(t):
    return np.where((2 < t) & (t < 2.39), -0.15, np.exp(-0.5 * t) * np.cos(3 * t) * np.cos(t))


def plot_optima_example():

    x = np.arange(0.0, 5.0, 0.01)

    plt.figure()
    plt.plot(x, f(x))
    plt.plot(0.87, -0.37, ".", color="red", markersize=10, lw=0)
    plt.plot(2.2, -0.16, "_", color="green", markersize=20, mew=5, lw=0, label="root")
    plt.plot(4, -0.08, ".", color="red", markersize=10, lw=0)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("x")
    plt.ylabel("f(x)")


def f(x):
    return np.cos(4 * x) * np.cos(0.7 * x)


def plot_true_observed_example():

    x = np.arange(0.0, 5.0, 0.01)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.plot(x, f(x), "black")
    ax1.plot(0.5, -0.5, ".", color="blue", markersize=12, lw=0)
    ax1.plot(0.5, -0.5, ".", color="blue", markersize=12, lw=0)
    ax1.plot(1.75, 0.25, ".", color="blue", markersize=12, lw=0)
    ax1.plot(1.95, 0.05, ".", color="blue", markersize=12, lw=0)
    ax1.plot(2.9, -0.25, ".", color="blue", markersize=12, lw=0)
    ax1.plot(3.8, 0.75, ".", color="blue", markersize=12, lw=0)
    ax1.plot(4.1, 0.8, ".", color="blue", markersize=12, lw=0)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlabel("x")
    ax1.set_ylabel("f(x)")
    ax1.set_title("True function")

    ax2.plot(x, f(x), alpha=0)
    ax2.plot(0.5, -0.5, ".", color="blue", markersize=12, lw=0)
    ax2.plot(0.5, -0.5, ".", color="blue", markersize=12, lw=0)
    ax2.plot(1.75, 0.25, ".", color="blue", markersize=12, lw=0)
    ax2.plot(1.95, 0.05, ".", color="blue", markersize=12, lw=0)
    ax2.plot(2.9, -0.25, ".", color="blue", markersize=12, lw=0)
    ax2.plot(3.8, 0.75, ".", color="blue", markersize=12, lw=0)
    ax2.plot(4.1, 0.8, ".", color="blue", markersize=12, lw=0)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlabel("x")
    ax2.set_ylabel("f(x)")
    ax2.set_title("Observed function")
