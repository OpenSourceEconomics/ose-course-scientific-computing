import matplotlib.pyplot as plt
import numpy as np


def plot_golden_search_problem(f):
    fig, ax = plt.subplots()
    grid = np.linspace(0, 3)
    vf = np.vectorize(f)
    ax.plot(grid, vf(grid))
