import numpy as np
from nonlinearity_utils import (
    return_nonlinear_fn,
    return_derivative_of_nonlinear_fn,
    return_nonlinear_fn_numpy,
)
from scipy.integrate import dblquad
from scipy.optimize import fsolve
from scipy.integrate import quad


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
    lambda_2 = 1
    lower = []
    snr = []
    noise_entropy = 1 / 2 * np.log(2 * np.pi * np.exp(1) * config["sigma_2"] ** 2)

    for lambda_2 in np.logspace(-1, 2, 20):
        fun1 = (
            lambda x, lambda_1: lambda_1 * (d_nonlinear(x)) * np.exp(-(x**2) / lambda_2)
        )
        fun2 = (
            lambda y, lambd: y**2.0
            * lambd
            * (d_nonlinear(y))
            * np.exp(-(y**2) / lambda_2)
        )

        fun3 = lambda lambda_1: quad(fun1, -np.inf, np.inf, args=(lambda_1))[0] - 1
        lambda_1 = fsolve(fun3, 1)[0]

        temp = quad(fun2, -np.inf, np.inf, args=(lambda_1))[0]
        entropy = -np.log(lambda_1) + (1 / (lambda_2)) * temp

        lower.append(
            1 / 2 * np.log(1 + np.exp(2 * entropy) / np.exp(2 * noise_entropy))
        )

        snr.append(10 * np.log10(temp / (config["sigma_2"] ** 2)))

    last_entropy = -np.log(1 / 2)
    last_lower = (
        1 / 2 * np.log(1 + np.exp(2 * last_entropy) / np.exp(2 * noise_entropy))
    )
    snr.append(40)
    lower.append(last_lower)
    return lower, snr


def upper_bound_tarokh(power, config):
    nonlinear_func = return_nonlinear_fn(config)
    upper_bound = (
        1 / 2 * np.log(1 + nonlinear_func(np.sqrt(power)) ** 2 / config["sigma_2"] ** 2)
    )
    return upper_bound


def lower_bound_tarokh_third_regime(power, config):
    nonlinear_func = return_nonlinear_fn(config)
    int_fz = return_integral_fz_for_tarokh_third_regime(power, config)
    f_term = 0.5 * np.log(
        2
        * np.pi
        * np.exp(1)
        * (
            (power + config["sigma_1"] ** 2) * np.exp(2 * int_fz)
            + config["sigma_2"] ** 2
        )
    )
    L = calculate_l_for_third_regime_lower_bound(power, config)
    s_term = -0.5 * np.log(2 * np.pi * np.exp(1) * (L + config["sigma_2"] ** 2))
    return f_term + s_term


def calculate_l_for_third_regime_lower_bound(power, config):
    nonlinear_func = return_nonlinear_fn_numpy(config)
    max_L = 0
    for epsilon in np.linspace(-1000, 1000, 200):
        fun = (
            lambda n: nonlinear_func(n) ** 2
            * 1
            / np.sqrt(2 * np.pi * config["sigma_1"] ** 2)
            * np.exp(-0.5 * (n - epsilon) ** 2 / config["sigma_1"] ** 2)
        )
        L = quad(fun, -np.inf, np.inf)[0]

        if L > max_L:
            max_L = L
    return max_L


def return_integral_fz_for_tarokh_third_regime(power, config):
    d_phi = return_derivative_of_nonlinear_fn(config)

    var_z = config["sigma_1"] ** 2 + power
    f_z = (
        lambda z: 1 / np.sqrt(2 * np.pi * var_z) * np.exp(-0.5 * (z) ** 2 / var_z)
    )  # chosen gaussian distribution with variance P + sigma_1^2 since X is Gaussian with variance P and mean 0

    new_fun = lambda z: np.log(d_phi(z) + 1e-20) * f_z(z)
    int_fz = quad(new_fun, -np.inf, np.inf)[0]
    return int_fz


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
