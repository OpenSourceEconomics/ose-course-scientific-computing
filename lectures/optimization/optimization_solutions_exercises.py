import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from optimization_problems import get_test_function
from scipy.optimize import minimize
from scipy.stats import logistic
from scipy.stats import norm


def test_exercise_1():
    a, b = 5, 0
    fvals = list()
    grid = np.linspace(-3, 4)
    for value in grid:
        fvals.append(get_test_function(value, a, b))
    plt.plot(grid, fvals)


def test_exercise_99():
    def binary_model(y, x, beta, distribution):
        F = distribution.cdf(x @ beta)
        fval = (y * np.log(F) + (1 - y) * np.log(1 - F)).sum()
        return -fval

    def logl_logit(y, x, beta):
        return binary_model(y, x, beta, logistic)

    def logl_probit(y, x, beta):
        return binary_model(y, x, beta, norm)

    dirname = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_pickle(f"{dirname}/material/data-graduation-prediction.pkl")
    x, y = df[["INTERCEPT", "GPA", "TUCE", "PSI"]], df["GRADE"]
    [minimize(model, [0.0] * 4, args=(y, x)) for model in [logl_logit, logl_probit]]
