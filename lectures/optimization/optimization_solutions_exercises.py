"""Solutions to exercises in optimization lab."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from labs.optimization.optimization_problems import get_test_function
from scipy.optimize import minimize
from scipy.stats import logistic
from scipy.stats import norm


def test_exercise_1():
    """Solution for exercise 1."""
    a, b = 5, 0
    fvals = []
    grid = np.linspace(-3, 4)
    for value in grid:
        fvals.append(get_test_function(value, a, b))
    plt.plot(grid, fvals)


def test_exercise_2():
    """Solution for exercise 2."""
    dirname = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_pickle(f"{dirname}/material/data-consumption-function.pkl")

    def construct_predicted_values(income, alpha, beta, gamma):
        return alpha + beta * income ** gamma

    mock_rslt = [-91.1933, 0.5691, 1.0204]
    income = df["realgdp"].values
    df["realcons_pred"] = construct_predicted_values(income, *mock_rslt)

    x = df.index.get_level_values("Year")
    fig, ax = plt.subplots()
    ax.plot(x, df["realcons_pred"], label="Predicted")
    ax.plot(x, df["realcons"], label="Observed")


def test_exercise_99():
    """Solution for exercise."""

    def binary_model(y, x, beta, distribution):
        """Get binary model."""
        F = distribution.cdf(x @ beta)
        fval = (y * np.log(F) + (1 - y) * np.log(1 - F)).sum()
        return -fval

    def logl_logit(y, x, beta):
        """Get logit model."""
        return binary_model(y, x, beta, logistic)

    def logl_probit(y, x, beta):
        """Get probit model."""
        return binary_model(y, x, beta, norm)

    dirname = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_pickle(f"{dirname}/material/data-graduation-prediction.pkl")
    x, y = df[["INTERCEPT", "GPA", "TUCE", "PSI"]], df["GRADE"]
    [minimize(model, [0.0] * 4, args=(y, x)) for model in [logl_logit, logl_probit]]
