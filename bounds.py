import numpy as np
from nonlinearity_utils import return_nonlinear_fn, return_derivative_of_nonlinear_fn
from scipy.integrate import dblquad


def bounds_l1_norm(power, sigma, config):
    # Only Nonlinearity is Linear
    if config["nonlinearity"] != 0:
        raise ValueError("Linear channel is required for this bound")

    upper_bound = np.log(
        2 * np.exp(1) * (power + sigma * np.sqrt(2 / np.pi))
    ) - 1 / 2 * np.log(2 * np.pi * np.exp(1) * sigma**2)
    lower_bound = 1 / 2 * np.log(1 + np.pi / 2 * power**2 / (sigma**2))
    return upper_bound, lower_bound


def lower_bound_tarokh(config):
    d_nonlinear = return_derivative_of_nonlinear_fn(config)

    # TODO: Implement this
    pass


def upper_bound_tarokh(power, config):
    nonlinear_func = return_nonlinear_fn(config)
    upper_bound = (
        1 / 2 * np.log(1 + nonlinear_func(np.sqrt(power)) ** 2 / config["sigma_2"] ** 2)
    )
    return upper_bound


def lower_bound_with_sdnr(power, config):
    nonlinear_func = return_nonlinear_fn(config)
    fun1 = (
        lambda x, z: x
        * nonlinear_func(x + z)
        * (1 / np.sqrt(2 * np.pi * power))
        * np.exp(-0.5 * x**2 / power)
        * (1 / (np.sqrt(2 * np.pi) * config["sigma_1"]))
        * np.exp(-0.5 * z**2 / (config["sigma_1"] ** 2))
    )
    gamma = dblquad(fun1, -np.inf, np.inf, -np.inf, np.inf)[
        0
    ]  # First one is the result, second one is the error

    fun2 = (
        lambda x, z: nonlinear_func(x + z)
        * nonlinear_func(x + z)
        * (1 / np.sqrt(2 * np.pi * power))
        * np.exp(-0.5 * x**2 / power)
        * (1 / (np.sqrt(2 * np.pi) * config["sigma_1"]))
        * np.exp(-0.5 * z**2 / (config["sigma_1"] ** 2))
    )
    sigma_y_2 = dblquad(fun2, -np.inf, np.inf, -np.inf, np.inf)[0]

    gamma = abs(gamma) ** 2 / (power * sigma_y_2)
    sdnr = gamma / (1 - gamma)

    return np.log(1 + sdnr) / 2
