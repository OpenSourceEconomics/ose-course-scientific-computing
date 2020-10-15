import numpy as np
import matplotlib.pyplot as plt


def plot_golden_search_problem(f, a, b):
    fig, ax = plt.subplots()
    grid = np.linspace(a, b)
    vf = np.vectorize(f)
    ax.plot(grid, vf(grid))
