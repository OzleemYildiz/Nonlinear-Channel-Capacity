import numpy as np
from nonlinearity_utils import (
    return_nonlinear_fn,
    return_derivative_of_nonlinear_fn,
    return_nonlinear_fn_numpy,
)
from scipy.integrate import dblquad, nquad
from scipy.optimize import fsolve
from scipy.integrate import quad
from matplotlib import pyplot as plt
from utils import (
    regime_dependent_snr,
    read_config,
    get_alphabet_x_y,
    get_regime_class,
    get_PP_complex_alphabet_x_y,
)
import math
from scipy.special import erfc, erf, comb
import torch
import numdifftools as nd
from sympy import Matrix, linsolve, symbols
from gaussian_capacity import get_gaussian_distribution


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
    d_nonlinear = return_derivative_of_nonlinear_fn(
        config, tanh_factor=config["tanh_factor"]
    )
    lambda_2 = 1
    lower = []
    snr = []

    noise_entropy = 1 / 2 * np.log(2 * np.pi * np.exp(1) * config["sigma_2"] ** 2)

    plt.figure(figsize=(5, 4))
    leg_str = []
    for lambda_2 in np.logspace(0, 4, 100):
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

        if config["complex"]:
            lower.append(np.log(1 + np.exp(2 * entropy) / np.exp(2 * noise_entropy)))
        else:
            lower.append(
                1 / 2 * np.log(1 + np.exp(2 * entropy) / np.exp(2 * noise_entropy))
            )
        # breakpoint()
        fx_x = lambda x: lambda_1 * (d_nonlinear(x)) * np.exp(-(x**2) / lambda_2)
        delta_x = np.linspace(-5, 5, 1000)[1] - np.linspace(-5, 5, 1000)[0]
        fx_x_list = [fx_x(x) * delta_x for x in np.linspace(-5, 5, 1000)]
        f_v = lambda v: fx_x(math.atanh(v)) * np.abs(1 / (1 - v**2))
        delta = np.linspace(-0.99, 0.99, 1000)[1] - np.linspace(-0.99, 0.99, 1000)[0]
        f_v_list = [f_v(v) * delta for v in np.linspace(-0.99, 0.99, 1000)]
        # print("Sum pdf:", np.sum(f_v_list))

        # gaus = (
        #     lambda x, sigma: 1
        #     / np.sqrt(2 * np.pi * sigma**2)
        #     * np.exp(-(x**2) / (2 * sigma**2))
        # )
        # gaus_list = [gaus(x, config["sigma_1"]) for x in np.linspace(-10, 10, 1000)]

        snr.append(10 * np.log10(temp / (config["sigma_2"] ** 2)))
    #     plt.plot(np.linspace(-0.99, 0.99, 1000), f_v_list)
    #     leg_str.append(str(snr[-1]))

    # plt.legend(leg_str)
    # plt.grid()
    # plt.show()
    # last_entropy = -np.log(1 / 2)
    # if config["complex"]:
    #     last_entropy = 2 * last_entropy
    #     last_lower = np.log(1 + np.exp(2 * last_entropy) / np.exp(2 * noise_entropy))
    # else:
    #     last_lower = (
    #         1 / 2 * np.log(1 + np.exp(2 * last_entropy) / np.exp(2 * noise_entropy))
    #     )
    snr.append(10 * np.log10(config["max_power_cons"] / (config["sigma_2"] ** 2)))
    lower.append(lower[-1])
    # lower.append(last_lower)

    return lower, snr


# Regime=1
def upper_bound_tarokh(power, config):
    nonlinear_func = return_nonlinear_fn(config, tanh_factor=config["tanh_factor"])
    upper_bound = (
        1 / 2 * np.log(1 + nonlinear_func(np.sqrt(power)) ** 2 / config["sigma_2"] ** 2)
    )
    if config["complex"] == True:
        upper_bound = 2 * upper_bound
    return upper_bound


def lower_bound_tarokh_third_regime(config):
    # snr_range = np.linspace(-20, 40, 20)
    snr_range, noise_power = regime_dependent_snr(config)
    # breakpoint()
    snr_range = np.concatenate(([-20, -10], snr_range))
    power_range = (10 ** (snr_range / 10)) * noise_power
    low_bound = []
    earlier_low = 0
    for power in power_range:
        nonlinear_func = return_nonlinear_fn(config, tanh_factor=config["tanh_factor"])
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
        # sprint(f_term + s_term)
        if f_term + s_term > earlier_low:
            earlier_low = f_term + s_term
            low_bound.append(f_term + s_term)
        else:
            low_bound.append(earlier_low)

    return low_bound[2:], snr_range[2:]


def lower_bound_tarokh_third_regime_with_pw(power, config):

    nonlinear_func = return_nonlinear_fn(config, tanh_factor=config["tanh_factor"])
    int_fz = return_integral_fz_for_tarokh_third_regime(power, config)
    print("Integral fz:", int_fz)
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
    print("L:", L)
    s_term = -0.5 * np.log(2 * np.pi * np.exp(1) * (L + config["sigma_2"] ** 2))

    return f_term + s_term


def calculate_l_for_third_regime_lower_bound(power, config):
    nonlinear_func = return_nonlinear_fn_numpy(
        config, tanh_factor=config["tanh_factor"]
    )
    max_L = 0
    for epsilon in np.linspace(-1000, 1000, 2001):
        fun = (
            lambda n: nonlinear_func(n) ** 2
            * 1
            / np.sqrt(2 * np.pi * config["sigma_1"] ** 2)
            * np.exp(-0.5 * (n - epsilon) ** 2 / config["sigma_1"] ** 2)
        )
        L = quad(fun, -np.inf, np.inf)[0] - expectation_of_phi_v(config, epsilon) ** 2
        if L > max_L:
            max_L = L
    return max_L


def expectation_of_phi_v(config, epsilon):
    nonlinear_func = return_nonlinear_fn_numpy(
        config, tanh_factor=config["tanh_factor"]
    )
    fun1 = (
        lambda z: nonlinear_func(z + epsilon)
        * 1
        / np.sqrt(2 * np.pi * config["sigma_1"] ** 2)
        * np.exp(-0.5 * (z) ** 2 / config["sigma_1"] ** 2)
    )
    int_fun1 = quad(fun1, -np.inf, np.inf)[0]
    return int_fun1


def return_integral_fz_for_tarokh_third_regime(power, config):
    d_phi = return_derivative_of_nonlinear_fn(config, tanh_factor=config["tanh_factor"])

    var_z = config["sigma_1"] ** 2 + power
    f_z = (
        lambda z: 1 / np.sqrt(2 * np.pi * var_z) * np.exp(-0.5 * (z) ** 2 / var_z)
    )  # chosen gaussian distribution with variance P + sigma_1^2 since X is Gaussian with variance P and mean 0

    new_fun = lambda z: np.log(d_phi(z) + 1e-20) * f_z(z)
    int_fz = quad(new_fun, -np.inf, np.inf)[0]
    return int_fz


def lower_bound_with_sdnr(power, config):
    nonlinear_func = return_nonlinear_fn(config, tanh_factor=config["tanh_factor"])
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


def simplified_first_from_third(power, config):
    d_phi = return_derivative_of_nonlinear_fn(config)
    phi = return_nonlinear_fn_numpy(config, tanh_factor=config["tanh_factor"])
    int_inside = (
        lambda x: 1
        / np.sqrt(2 * np.pi * power)
        * np.exp(-0.5 * (x) ** 2 / power)
        * np.log(d_phi(x) + 1e-20)
    )
    int_out = quad(int_inside, -np.inf, np.inf)[0]
    # breakpoint()
    low = 0.5 * np.log(1 + (power * np.exp(2 * int_out)) / config["sigma_2"] ** 2)
    return low


def main():
    config = read_config()
    print("----------Nonlinearity: ", config["nonlinearity"], "----------")
    snr_change, noise_power = regime_dependent_snr(config)

    # my_first_tarokh = lower_bound_tarokh(config)
    my_third_tarokh = []
    # simplified_first_f_third = []
    logsnr = []
    snr_change = np.linspace(-20, 40, 50)
    earlier_calc = 0
    for snr in snr_change:
        print("--------", snr, "------------")
        power = (10 ** (snr / 10)) * noise_power
        find_lambda1_lambda2_for_dist(power, config)
        upper_bound_tarokh_third_regime(power, config)
        sundeep_upper_bound_third_regime(power, config)

        calc = lower_bound_tarokh_third_regime_with_pw(power, config)
        if calc > earlier_calc:
            earlier_calc = calc
            my_third_tarokh.append(calc)
        else:
            my_third_tarokh.append(earlier_calc)
        logsnr.append(np.log(1 + power / (noise_power)) / 2)
        # simplified_first_f_third.append(simplified_first_from_third(power, config))
    # breakpoint()
    plt.figure()
    # plt.plot(my_first_tarokh[1], my_first_tarokh[0], label="First Regime")
    # plt.plot(snr_change, logsnr, label="Without Nonlinearity")
    plt.plot(snr_change, my_third_tarokh, label="Third Regime")
    # plt.plot(snr_change, simplified_first_f_third, label="Simplified First from Third")
    plt.legend()
    plt.show()


# clipping function - regime 1 - average power constraint
# call in the for loop for every power value
# - This does not make sense if it's real signal
def sdnr_bound_regime_1_tarokh_ref7(power, config):
    A = config["clipping_limit_x"]

    Omega = power  # average power
    gamma = A / np.sqrt(Omega)
    alpha = 1 - np.exp(-(gamma**2)) + np.sqrt(np.pi) / 2 * gamma * erfc(gamma)
    sigma_d_2 = Omega * (1 - np.exp(-(gamma**2)) - alpha**2)
    cap = np.log(1 + alpha**2 * Omega / (sigma_d_2 + config["sigma_2"] ** 2))
    return cap


# The following checks if lowering the power increases the capacity
# - This does not make sense if it's real signal
def updated_sdnr_bound_regime_1_tarokh_ref7(power, config):
    A = config["clipping_limit_x"]
    max_pow = power
    list_pow = np.linspace(max_pow, 0.01, 1000)
    best_cap = 0
    list_pow = [power]
    for power in list_pow:

        Omega = power  # average power
        gamma = A / np.sqrt(Omega)
        alpha = 1 - np.exp(-(gamma**2)) + np.sqrt(np.pi) / 2 * gamma * erfc(gamma)
        alpha = alpha
        sigma_d_2 = Omega * (1 - np.exp(-(gamma**2)) - alpha**2)

        cap = (
            1 / 2 * np.log(1 + alpha**2 * Omega / (sigma_d_2 + config["sigma_2"] ** 2))
        )
        if cap >= best_cap:
            best_cap = cap
        else:
            break

    return cap


def sdnr_new(power, config):

    max_pow = power
    list_pow = np.linspace(max_pow, 0.01, 1000)
    best_cap = 0
    for power in list_pow:
        Omega = power  # average power
        gaus_pdf = (
            lambda x: 1 / np.sqrt(2 * np.pi * Omega) * np.exp(-0.5 * x**2 / Omega)
        )  # Gaussian pdf with variance Omega

        b_calc = (
            lambda x: config["clipping_limit_y"]
            / config["clipping_limit_x"]
            * gaus_pdf(x)
        )
        B = quad(
            b_calc,
            -config["clipping_limit_x"],
            config["clipping_limit_x"],
        )[0]

        # print("B:", B)
        nonlin_fn = return_nonlinear_fn(config, tanh_factor=config["tanh_factor"])
        z_calc = lambda x: (nonlin_fn(x)) ** 2 * gaus_pdf(x)
        e_z2 = quad(
            z_calc,
            -config["clipping_limit_x"],
            config["clipping_limit_x"],
        )[0]

        z_calc = lambda x: (config["clipping_limit_y"]) ** 2 * gaus_pdf(x)
        e_z2 += quad(z_calc, -np.inf, -config["clipping_limit_x"])[0]
        e_z2 += quad(z_calc, config["clipping_limit_x"], np.inf)[0]

        # print("Ez2:", e_z2)
        sigma_d_2 = e_z2 - B**2 * power
        if sigma_d_2 > 0:
            cap = (
                1 / 2 * np.log(1 + B**2 * power / (sigma_d_2 + config["sigma_2"] ** 2))
            )
            if np.isnan(cap):
                breakpoint()

            if cap >= best_cap:
                best_cap = cap
            else:
                break
    return best_cap


def sdnr_new_with_erf(power, config):
    max_pow = power
    list_pow = np.linspace(max_pow, 0.01, 1000)
    best_cap = 0

    for power in list_pow:
        Omega = power  # average power
        gaus_pdf = (
            lambda x: 1 / np.sqrt(2 * np.pi * Omega) * np.exp(-0.5 * x**2 / Omega)
        )  # Gaussian pdf with variance Omega

        B = (
            config["clipping_limit_y"]
            / config["clipping_limit_x"]
            * erf(config["clipping_limit_x"] / np.sqrt(2 * Omega))
        )

        z_calc = lambda x: x**2 * gaus_pdf(x)
        A = quad(z_calc, 0, config["clipping_limit_x"])[0]
        C_z = 2 * (
            config["clipping_limit_y"] / config["clipping_limit_x"]
        ) ** 2 * A + config["clipping_limit_y"] ** 2 * (
            1 - erf(config["clipping_limit_x"] / np.sqrt(2 * Omega))
        )

        sigma_d_2 = C_z - B**2 * power
        cap = 1 / 2 * np.log(1 + B**2 * power / (sigma_d_2 + config["sigma_2"] ** 2))
        if np.isnan(cap):
            breakpoint()

        if cap >= best_cap:
            best_cap = cap
        else:
            break
    return best_cap


def sdnr_new_with_erf_nopowchange(power, config):

    Omega = power  # average power
    gaus_pdf = (
        lambda x: 1 / np.sqrt(2 * np.pi * Omega) * np.exp(-0.5 * x**2 / Omega)
    )  # Gaussian pdf with variance Omega

    B = (
        config["clipping_limit_y"]
        / config["clipping_limit_x"]
        * erf(config["clipping_limit_x"] / np.sqrt(2 * Omega))
    )

    z_calc = lambda x: x**2 * gaus_pdf(x)
    A = quad(z_calc, 0, config["clipping_limit_x"])[0]
    C_z = 2 * (
        config["clipping_limit_y"] / config["clipping_limit_x"]
    ) ** 2 * A + config["clipping_limit_y"] ** 2 * (
        1 - erf(config["clipping_limit_x"] / np.sqrt(2 * Omega))
    )

    sigma_d_2 = C_z - B**2 * power
    cap = 1 / 2 * np.log(1 + B**2 * power / (sigma_d_2 + config["sigma_2"] ** 2))
    if config["complex"]:
        cap = 2 * cap
    if np.isnan(cap):
        breakpoint()
    return cap


def sdnr_new_rayleigh(power, config):

    max_pow = power
    list_pow = np.linspace(max_pow, 0.01, 1000)
    best_cap = 0
    func = return_nonlinear_fn(config, tanh_factor=config["tanh_factor"])
    for power in list_pow:
        # f_r = lambda r: (  # <- Full Power
        #     2 * r / (power * 2) * np.exp(-(r**2) / (power * 2)) if r >= 0 else 0
        # )
        # # f_r = lambda r: (
        # #     2 * r / power * np.exp(-(r**2) / power) if r >= 0 else 0
        # # )  # <- Half Power

        # # # uniform distribution over [0, 2pi]
        # f_theta = lambda th: 1 / (2 * np.pi) if th >= 0 and th <= 2 * np.pi else 0
        # hold = lambda r, th: r * func(r) * np.cos(th) ** 2 * f_r(r) * f_theta(th)
        # E_XY = nquad(
        #     lambda r, th: r * func(r) * np.cos(th) ** 2 * f_r(r) * f_theta(th),
        #     [[0, np.inf], [0, 2 * np.pi]],
        # )[0]

        # E_XX = nquad(
        #     lambda r, th: r**2 * np.cos(th) ** 2 * f_r(r) * f_theta(th),
        #     [[0, np.inf], [0, 2 * np.pi]],
        # )[0]

        # alpha = E_XY / (E_XX)

        # gamma = config["clipping_limit_x"] / np.sqrt(power)
        gamma = config["clipping_limit_x"] / np.sqrt(
            power * 2
        )  # <- Correct Power in Real
        alpha2 = 1 - np.exp(-(gamma**2)) + np.sqrt(np.pi) / 2 * gamma * erfc(gamma)

        # E_YY = nquad(
        #     lambda r, th: func(r) ** 2 * np.cos(th) ** 2 * f_r(r) * f_theta(th),
        #     [[0, np.inf], [0, 2 * np.pi]],
        # )[0]

        # E_DD = E_YY - alpha**2 * E_XX
        # cap = 1 / 2 * np.log(1 + alpha**2 * power / (E_DD + config["sigma_2"] ** 2))

        sigma_d_2 = power * (1 - np.exp(-(gamma**2)) - alpha2**2)
        cap = (
            1 / 2 * np.log(1 + alpha2**2 * power / (sigma_d_2 + config["sigma_2"] ** 2))
        )
        # breakpoint()
        # breakpoint()
        # r_list = np.linspace(0, 20, 1000)
        # f_gr_list = [f_gr(r) for r in r_list]
        # theta = np.linspace(0, 2 * np.pi, 1000)
        # f_theta_list = [f_theta(th) for th in theta]
        # f_theta_list = np.array(f_theta_list)
        # alph = np.array([])
        # pdf = np.array([])
        # for r in r_list:
        #     alph = np.append(alph, func(r) * np.sin(theta))
        #     pdf = np.append(pdf, f_gr(r) * f_theta_list)
        # plt.scatter(alph, pdf)
        # plt.show()
        # breakpoint()
        if best_cap < cap:
            best_cap = cap
        else:
            break
    return best_cap

    # return best_cap


# This is a bound for the third regime - At least as a first try
# First is to try with the clip function
# I will take X - Clipped Gaussian
#             W_1 -Gaussian


# Y = phi(X+W_1)+W_2
def sundeep_upper_bound_third_regime(power, config):
    func = return_nonlinear_fn(config, tanh_factor=config["tanh_factor"])
    expectation_X = 0
    alphabet_x, alphabet_y, max_x, max_y = get_alphabet_x_y(config, power)
    delta_x = alphabet_x[1] - alphabet_x[0]
    max_W_1 = config["sigma_1"] * config["stop_sd"]
    alphabet_w1 = torch.arange(-max_W_1, max_W_1 + delta_x / 2, delta_x)
    pdf_w1 = (
        1
        / (torch.sqrt(torch.tensor([2 * torch.pi * config["sigma_1"] ** 2])))
        * torch.exp(-0.5 * ((alphabet_w1) ** 2) / config["sigma_1"] ** 2).float()
    )
    pdf_w1 = (pdf_w1 / torch.sum(pdf_w1)).to(torch.float32)

    pdf_x = (
        1
        / (torch.sqrt(torch.tensor([2 * torch.pi * power])))
        * torch.exp(-0.5 * ((alphabet_x) ** 2) / power).float()
    )
    pdf_x = (pdf_x / torch.sum(pdf_x)).to(torch.float32)
    inside = (
        func(alphabet_x.reshape(-1, 1) + alphabet_w1.reshape(1, -1))
        - func(expectation_X + alphabet_w1.reshape(1, -1))
    ) ** 2 / (alphabet_x.reshape(-1, 1) - expectation_X) ** 2
    c_a = pdf_x @ inside

    # print("+Power:", power)
    # print("E_x:", (alphabet_x**2) @ pdf_x)

    E_x = (alphabet_x**2) @ pdf_x  # energy per symbol
    mut_info_a = torch.log(1 + c_a * E_x / (config["sigma_2"] ** 2))
    cap = mut_info_a @ pdf_w1
    print("Sundeep Upper Bound:", cap)
    return cap


def upper_bound_tarokh_third_regime(power, config):
    func = return_nonlinear_fn(config, tanh_factor=config["tanh_factor"])
    d_func = return_derivative_of_nonlinear_fn(
        config, tanh_factor=config["tanh_factor"]
    )
    alphabet_x, alphabet_y, max_x, max_y = get_alphabet_x_y(
        config, power, tanh_factor=config["tanh_factor"]
    )
    alphabet_x = alphabet_x.numpy()
    sum_bottom = 0
    pdf_x = (
        1 / (np.sqrt(2 * torch.pi * power)) * np.exp(-0.5 * ((alphabet_x) ** 2) / power)
    )
    pdf_x = (pdf_x / np.sum(pdf_x)).astype(np.float32)

    for ind, x in enumerate(alphabet_x):
        f_z = (
            lambda z: 1
            / np.sqrt(2 * np.pi * config["sigma_1"] ** 2)
            * np.exp(-0.5 * (z - x) ** 2 / config["sigma_1"] ** 2)
        )
        func = lambda z: f_z(z) * np.log(d_func(z) + 1e-20)
        int1 = quad(func, -np.inf, np.inf)[0]
        sum_bottom += (
            config["sigma_1"] ** 2 * np.exp(2 * int1) + config["sigma_2"] ** 2
        ) * pdf_x[ind]
    # breakpoint()

    not_stop = True
    k = 1
    sum_top = 0
    while not_stop:
        earlier_top = sum_top
        for i in range(0, 2 * k + 1):
            # breakpoint()
            phi_i = nd.Derivative(func, n=i)
            phi_ki = nd.Derivative(func, n=2 * k - i)
            sum_top += (
                comb(2 * k, i)
                * phi_i(np.sqrt(power))
                * phi_ki(np.sqrt(power))
                * config["sigma_1"] ** (2 * k)
                / (2 * k)
            )
        k += 1
        if abs(sum_top - earlier_top) < 1e-5:
            not_stop = False
        # if k > 4:  # FIXME: better method is needed
        #     not_stop = False

    upper_cap = (
        1
        / 2
        * np.log(
            1
            + (func(np.sqrt(power)) ** 2 + sum_top + config["sigma_2"] ** 2)
            / sum_bottom
        )
    )

    # Updated because it does not work well for low SNR
    # upper_cap = min(
    #     upper_cap,
    #     0.5 * np.log(1 + power / (config["sigma_2"] ** 2 + config["sigma_1"] ** 2)),
    # )

    print("Tarokh Upper Bound:", upper_cap)
    return upper_cap


def lower_bound_by_mmse_correlation(power, config):
    func = return_nonlinear_fn(config, tanh_factor=config["tanh_factor"])

    pdf_x = lambda x: (1 / (np.sqrt(2 * np.pi * power)) * np.exp(-0.5 * (x**2) / power))
    E_Y_f = lambda x: func(x) * pdf_x(x)
    E_Y = quad(E_Y_f, -np.inf, np.inf)[0]
    E_Y2_f = lambda x: func(x) ** 2 * pdf_x(x)
    E_Y2 = quad(E_Y2_f, -np.inf, np.inf)[0] + config["sigma_2"] ** 2
    var_Y = E_Y2 - E_Y**2
    var_X = power
    E_X = 0
    E_XY_f = lambda x: x * func(x) * pdf_x(x)
    E_XY = quad(E_XY_f, -np.inf, np.inf)[0]
    cov_XY = E_XY - E_X * E_Y
    correlation = cov_XY / (np.sqrt(var_X) * np.sqrt(var_Y))
    lower_bound = -0.5 * np.log(1 - correlation**2)

    return lower_bound


def lower_bound_by_mmse_correlation_numerical(power, config):
    func = return_nonlinear_fn_numpy(config, tanh_factor=config["tanh_factor"])

    alphabet_x, alphabet_y, max_x, max_y = get_alphabet_x_y(config, power)

    alphabet_x = alphabet_x.numpy()
    alphabet_y = alphabet_y.numpy()
    pdf_x = 1 / np.sqrt(2 * np.pi * power) * np.exp(-0.5 * (alphabet_x**2) / power)
    pdf_x = pdf_x / np.sum(pdf_x)
    E_Y = func(alphabet_x) @ pdf_x
    E_Y2 = (func(alphabet_x) ** 2) @ pdf_x + config["sigma_2"] ** 2

    var_Y = E_Y2 - E_Y**2
    var_X = power
    E_X = 0

    E_XY = (alphabet_x * func(alphabet_x)) @ pdf_x

    cov_XY = E_XY - E_X * E_Y
    correlation = cov_XY / (np.sqrt(var_X) * np.sqrt(var_Y))
    lower_bound = -0.5 * np.log(1 - correlation**2)
    if np.isnan(lower_bound):
        breakpoint()

    return lower_bound


def lower_bound_by_mmse(power, config):
    func = return_nonlinear_fn_numpy(config, tanh_factor=config["tanh_factor"])

    alphabet_x, alphabet_y, max_x, max_y = get_alphabet_x_y(config, power)
    alphabet_x = alphabet_x.numpy()
    alphabet_y = alphabet_y.numpy()

    pdf_x = 1 / (np.sqrt(2 * np.pi * power)) * np.exp(-0.5 * (alphabet_x**2) / power)
    pdf_x = pdf_x / np.sum(pdf_x)
    pdf_y_given_x = (
        1
        / (np.sqrt(2 * np.pi * config["sigma_2"] ** 2))
        * np.exp(
            -0.5
            * (alphabet_y.reshape(-1, 1) - func(alphabet_x).reshape(1, -1)) ** 2
            / config["sigma_2"] ** 2
        )
    )
    pdf_y_given_x = pdf_y_given_x / (np.sum(pdf_y_given_x, axis=0) + 1e-30)

    pdf_y = pdf_y_given_x @ pdf_x
    pdf_y = pdf_y / np.sum(pdf_y)

    pdf_x_given_y = (
        pdf_y_given_x * pdf_x.reshape(1, -1) / (pdf_y.reshape(-1, 1) + 1e-30)
    )
    pdf_x_given_y = pdf_x_given_y.transpose()
    pdf_x_given_y = pdf_x_given_y / (np.sum(pdf_x_given_y, axis=0) + 1e-30)

    E_X_given_Y = alphabet_x @ pdf_x_given_y

    E_X2_given_y = (alphabet_x**2) @ pdf_x_given_y

    E_Y = func(alphabet_x) @ pdf_x
    E_Y2 = (func(alphabet_x) ** 2) @ pdf_x + config["sigma_2"] ** 2
    var_Y = E_Y2 - E_Y**2
    var_X = power
    E_X = 0
    E_XY = (alphabet_x * func(alphabet_x)) @ pdf_x

    cov_XY = E_XY - E_X * E_Y
    alpha = cov_XY / var_Y
    distortion = E_X - alpha * E_Y

    sigma_y_2 = (
        E_X2_given_y
        - 2 * E_X_given_Y * (alpha * alphabet_y + distortion)
        + (alpha * alphabet_y + distortion) ** 2
    )

    upper_H_X_given_Y = 0.5 * np.log(2 * np.pi * np.exp(1) * sigma_y_2) @ pdf_y

    # H(X) = 1/2*ln(2*pi*e*sigma_x^2)
    # H(X|Y) = 1/2*E[ln(2*pi*e*sigma_y^2)]
    # H(X) - H(X|Y) = I(X;Y)
    lower_bound = 0.5 * np.log(2 * np.pi * np.exp(1) * power) - upper_H_X_given_Y

    # phi(X)+N =Y
    if np.isnan(lower_bound):
        breakpoint()

    return lower_bound


def lower_bound_by_mmse_with_truncated_gaussian(power, config):
    func = return_nonlinear_fn(config, tanh_factor=config["tanh_factor"])
    max_lower_bound = 0

    # power decrease might be necessary (similar to SDNR)
    power_range = np.linspace(power, 0.01, 100)
    power_range = [power]
    for p in power_range:
        # psi_x = lambda x: 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2)
        # phi_x = lambda x: 1 / 2 * (1 + erf(x / np.sqrt(2)))
        # pdf_x = lambda x: (
        #     1
        #     / np.sqrt(p)
        #     * psi_x(x / np.sqrt(p))
        #     / (
        #         phi_x(config["clipping_limit_x"] / np.sqrt(p))
        #         - phi_x(-config["clipping_limit_x"] / np.sqrt(p))
        #     )
        #     if x < config["clipping_limit_x"] and x > -config["clipping_limit_x"]
        #     else 0
        # )
        # p = 100
        remaining = 1 / 2 * (1 - erf(config["clipping_limit_x"] / np.sqrt(2 * p)))
        pdf_x = lambda x: (
            1 / (np.sqrt(2 * np.pi * p)) * np.exp(-0.5 * (x**2) / p)
            if x < config["clipping_limit_x"] and x > -config["clipping_limit_x"]
            else (
                remaining
                if x == config["clipping_limit_x"] or x == -config["clipping_limit_x"]
                else 0
            )
        )
        # xxx = np.linspace(-20, 20, 1000)
        # xxx = np.append(xxx, [10, -10])
        # tt = [pdf_x(x) for x in xxx]
        #
        E_Y_f = lambda x: func(x) * pdf_x(x)
        E_Y = quad(E_Y_f, -np.inf, np.inf)[0]
        E_Y += E_Y_f(config["clipping_limit_x"]) + E_Y_f(
            -config["clipping_limit_x"]
        )  # The limit points are discrete distributions

        E_Y2_f = lambda x: func(x) ** 2 * pdf_x(x)
        E_Y2 = quad(E_Y2_f, -np.inf, np.inf)[0] + config["sigma_2"] ** 2
        E_Y2 += E_Y2_f(config["clipping_limit_x"]) + E_Y2_f(-config["clipping_limit_x"])

        var_Y = E_Y2 - E_Y**2
        var_X = p
        E_X = 0
        E_XY_f = lambda x: x * func(x) * pdf_x(x)
        E_XY = quad(E_XY_f, -np.inf, np.inf)[0]
        E_XY += E_XY_f(config["clipping_limit_x"]) + E_XY_f(-config["clipping_limit_x"])

        cov_XY = E_XY - E_X * E_Y
        correlation = cov_XY / (np.sqrt(var_X) * np.sqrt(var_Y))
        lower_bound = -0.5 * np.log(1 - correlation**2)
        if max_lower_bound < lower_bound:
            max_lower_bound = lower_bound

        else:
            break

    return max_lower_bound


# Prob. Distribution f_x(x) = lambda_1*phi_d(x)*exp(-x^2*lambda_2)
def find_lambda1_lambda2_for_dist(power, config):
    phi = return_nonlinear_fn(config, tanh_factor=config["tanh_factor"])
    phi_d = nd.Derivative(phi, n=1)
    breakpoint()

    pass


# def reg_mmse_bound(power, config):
#     func = return_nonlinear_fn(config)
#     max_lower_bound = 0

#     # power decrease might be necessary (similar to SDNR)
#     # power_range = np.linspace(power, 0.01, 100)
#     power_range = [power]
#     for p in power_range:
#         # I need to calculate EX_given_Y
#         pdf_x = lambda x: (1 / (np.sqrt(2 * np.pi * p)) * np.exp(-0.5 * (x**2) / p))

#         pdf_y_given_x = lambda y, x: (
#             1
#             / (np.sqrt(2 * np.pi * config["sigma_2"] ** 2))
#             * np.exp(-0.5 * (y - func(x)) ** 2 / config["sigma_2"] ** 2)
#         )
#         pdf_y = lambda y: quad(
#             lambda x: pdf_y_given_x(y, x) * pdf_x(x), -np.inf, np.inf
#         )[0]
#         pdf_x_given_y = lambda x, y: pdf_y_given_x(y, x) * pdf_x(x) / pdf_y(y)
#         # breakpoint()
#         E_X_given_Y = lambda y: quad(
#             lambda x: x * pdf_x_given_y(x, y), -np.inf, np.inf
#         )[0]
#         # E_X_hat = quad(lambda y: E_X_given_Y(y) * pdf_y(y), -np.inf, np.inf)[0]
#         # breakpoint()
#         E_X_Xhat = nquad(
#             lambda x, y: x * E_X_given_Y(y) * pdf_x(x) * pdf_y(y),
#             [[-np.inf, np.inf], [-np.inf, np.inf]],
#         )[0]
#         # breakpoint()
#         E_Xhat_2 = quad(lambda y: E_X_given_Y(y) ** 2 * pdf_y(y), -np.inf, np.inf)[0]

#         mse = p - 2 * E_X_Xhat + E_Xhat_2
#         # breakpoint()

#         lower_bound = -0.5 * np.log(1 - E_Xhat_2 / (mse + config["sigma_2"] ** 2))

#         # phi(X)+N =Y
#         # X' = E[X|Y]

#         print("&&&--MSE:", lower_bound)
#         if max_lower_bound < lower_bound:
#             max_lower_bound = lower_bound

#         else:
#             break

#     return max_lower_bound


def reg_mmse_bound_numerical(power, config):
    func = return_nonlinear_fn_numpy(config, tanh_factor=config["tanh_factor"])

    if config["complex"]:
        real_x, imag_x, real_y, imag_y = get_PP_complex_alphabet_x_y(
            config, power, config["tanh_factor"]
        )
        regime_class = get_regime_class(
            config=config,
            alphabet_x=real_x,
            alphabet_y=real_y,
            power=power,
            tanh_factor=config["tanh_factor"],
            alphabet_x_imag=imag_x,
            alphabet_y_imag=imag_y,
        )
    else:
        alphabet_x, alphabet_y, max_x, max_y = get_alphabet_x_y(
            config, power, tanh_factor=config["tanh_factor"], bound=True
        )
        regime_class = get_regime_class(
            config, alphabet_x, alphabet_y, power, config["tanh_factor"]
        )
    pdf_x = get_gaussian_distribution(
        power, regime_class, complex_alphabet=config["complex"]
    )

    alphabet_x = regime_class.alphabet_x.numpy()
    alphabet_y = regime_class.alphabet_y.numpy()

    pdf_x = pdf_x.numpy() / np.sum(pdf_x.numpy())
    if config["complex"]:
        out_nonliear = func(abs(alphabet_x)) * np.exp(1j * np.angle(alphabet_x))
        pdf_y_given_x = (
            1
            / (np.pi * config["sigma_2"] ** 2)
            * np.exp(
                -1
                * (np.abs(alphabet_y.reshape(-1, 1) - out_nonliear.reshape(1, -1))) ** 2
                / config["sigma_2"] ** 2
            )
        )
    else:
        pdf_y_given_x = (
            1
            / (np.sqrt(2 * np.pi * config["sigma_2"] ** 2))
            * np.exp(
                -0.5
                * (alphabet_y.reshape(-1, 1) - func(alphabet_x).reshape(1, -1)) ** 2
                / config["sigma_2"] ** 2
            )
        )
    pdf_y_given_x = pdf_y_given_x / (np.sum(pdf_y_given_x, axis=0) + 1e-30)

    pdf_y = pdf_y_given_x @ pdf_x
    pdf_y = pdf_y / np.sum(pdf_y)

    pdf_x_given_y = (
        pdf_y_given_x * pdf_x.reshape(1, -1) / (pdf_y.reshape(-1, 1) + 1e-30)
    )
    pdf_x_given_y = pdf_x_given_y.transpose()
    pdf_x_given_y = pdf_x_given_y / (np.sum(pdf_x_given_y, axis=0) + 1e-30)
    # breakpoint()
    E_X_given_Y = alphabet_x @ pdf_x_given_y

    E_X2_given_y = (abs(alphabet_x) ** 2) @ pdf_x_given_y

    # sigma_y_2 = E_X2_given_y - 2 * E_X_Xhat_given_y + E_Xhat_2_given_y
    sigma_y_2 = E_X2_given_y - abs(E_X_given_Y) ** 2

    upper_H_X_given_Y = 0.5 * np.log(2 * np.pi * np.exp(1) * sigma_y_2 + 1e-30) @ pdf_y
    if config["complex"]:
        upper_H_X_given_Y = 2 * upper_H_X_given_Y

    # mse = p - 2 * E_X_Xhat + E_Xhat_2
    # H(X) = E[ln(2*pi*e*sigma_x^2)]
    # H(X|Y) = E[ln(2*pi*e*sigma_y^2)]
    # H(X) - H(X|Y) = I(X;Y)
    H_X = 0.5 * np.log(2 * np.pi * np.exp(1) * power)
    if config["complex"]:
        H_X = 2 * H_X
    lower_bound = H_X - upper_H_X_given_Y
    # phi(X)+N =Y
    # X' = E[X|Y]
    if np.isnan(lower_bound):
        breakpoint()

    return lower_bound


def upper_bound_peak(power, config):
    if not config["nonlinearity"] == 5:
        raise ValueError("Nonlinearity should be 5")
    if config["regime"] == 3:
        # While it's before clip: Y = X+Z_1+Z_2
        snr = power / (config["sigma_2"] ** 2 + config["sigma_1"] ** 2)
        snr_peak = config["clipping_limit_x"] ** 2 / (
            config["sigma_2"] ** 2 + config["sigma_1"] ** 2
        )
    elif config["regime"] == 1:  # Y= X+Z_2
        snr = power / config["sigma_2"] ** 2
        snr_peak = config["clipping_limit_x"] ** 2 / config["sigma_2"] ** 2
    else:
        raise ValueError("Regime not defined")

    peak_cap = np.log(
        1
        + np.sqrt(
            2
            * (config["clipping_limit_x"] / config["sigma_2"]) ** 2
            / (np.pi * np.exp(1))
        )
    )

    pe = 1 / 2 * np.log(1 + snr_peak)  # General

    linear_cap = 1 / 2 * np.log(1 + snr)
    if config["complex"]:
        cap = min(2 * pe, 2 * linear_cap)
    else:
        cap = min(peak_cap, linear_cap)
    # breakpoint()
    return cap


def upper_bound_peak_power(x_c):
    return 2 * x_c**2 / (np.pi * np.exp(1)) + 2 * x_c * np.sqrt(2) / np.sqrt(
        np.pi * np.exp(1)
    )


def bound_backtracing_check(earlier_list, new_input):
    if len(earlier_list) == 0:
        earlier_list.append(new_input)
    elif new_input > earlier_list[-1]:
        earlier_list.append(new_input)
    else:
        earlier_list.append(earlier_list[-1])
    return earlier_list


def linear_interference_res(power1, power2, config):
    # Y1 = X1 +aX2+ Z11 + Z12
    # Y2 = X2 + Z21 + Z22

    if config["regime"] == 3:
        # Treat Interference as Noise
        tin_R1 = (
            1
            / 2
            * np.log(
                1
                + power1
                / (
                    config["sigma_12"] ** 2
                    + config["sigma_11"] ** 2
                    + config["int_ratio"] ** 2 * power2
                )
            )
        )

        # Known Interference
        ki_R1 = (
            1
            / 2
            * np.log(1 + power1 / (config["sigma_12"] ** 2 + config["sigma_11"] ** 2))
        )
        R2 = (
            1
            / 2
            * np.log(1 + power2 / (config["sigma_21"] ** 2 + config["sigma_22"] ** 2))
        )
        return tin_R1, ki_R1, R2
    else:
        raise ValueError("Regime not defined")


if __name__ == "__main__":
    main()
