import numpy as np
from scipy.optimize import minimize
from scipy.stats import logistic
from scipy.stats import norm


def test_exercise_1():
    def criterion(x):
        # inserted budget constraint
        return np.sqrt(x) + 2 * np.sqrt(1 - 0.33 * x)


def test_exercise_99():
    def binary_model(beta, distribution):
        F = distribution.cdf(x @ beta)
        fval = (y * np.log(F) + (1 - y) * np.log(1 - F)).sum()
        return -fval

    def logl_logit(beta):
        return binary_model(beta, logistic)

    def logl_probit(beta):
        return binary_model(beta, norm)

    for model in [logl_logit, logl_probit]:
        minimize(model, [0.0] * 4)
