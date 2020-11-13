import numpy as np
import scipy as sp


def golden_search_problem(x):
    return x * np.cos(x ** 2)


def get_nelder_mead_problem(x):
    # TODO: in progress of integrated in temfpy
    return sp.optimize.rosen(x)


def _get_test_function_gradient(x, ill_conditioned, add_noise):
    dimension = x.shape[0]
    quadratic_coeff, noise_coeff = _get_tuning_parameters(ill_conditioned, add_noise, dimension)

    return np.array(
        np.multiply(quadratic_coeff, np.array(x) - np.ones(np.array(x).size))
    ) + noise_coeff * 2 * np.pi * np.array(
        np.sin(2 * np.pi * (np.array(x) - np.ones(np.array(x).size)))
    )


def _get_tuning_parameters(ill_conditioned, add_noise, dimension):
    np.random.seed(123)
    conditioning_factor = float(ill_conditioned) * 20
    noise_coeff = 0.5 * float(add_noise)
    quadratic_coeff = np.array(np.exp(np.random.random(dimension) * conditioning_factor))
    quadratic_coeff = quadratic_coeff / np.max(quadratic_coeff)

    return quadratic_coeff, noise_coeff


def get_test_function(x, ill_conditioned=True, add_noise=True):
    def _fval(x):
        return (
            0.5
            * np.sum(
                np.multiply(quadratic_coeff, np.square(np.array(x) - np.ones(np.array(x).size)))
            )
            + dimension * noise_coeff
            - noise_coeff * np.sum(np.cos(2 * np.pi * (np.array(x) - np.ones(np.array(x).size))))
        )

    def _fhess(x, quadratic_coeff, noise_coeff):
        return np.diag(quadratic_coeff) + noise_coeff * 4 * np.square(np.pi) * np.diag(
            np.cos(2 * np.pi * (np.array(x) - np.ones(np.array(x).size)))
        )

    x = np.atleast_1d(x)

    dimension = x.shape[0]
    quadratic_coeff, noise_coeff = _get_tuning_parameters(ill_conditioned, add_noise, dimension)
    return _fval(x)  # , _get_test_function_gradient(x, ill_conditioned, add_noise), _fhess(x,
    # quadratic_coeff, noise_coeff)
