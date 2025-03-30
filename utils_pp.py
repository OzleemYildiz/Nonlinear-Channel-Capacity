import numpy as np
from gaussian_capacity import gaussian_capacity
from utils import (
    get_alphabet_x_y,
    get_PP_complex_alphabet_x_y,
    get_regime_class,
    grid_minor,
)
from bounds import reg_mmse_bound_numerical, bound_backtracing_check
import matplotlib.pyplot as plt
from gd import gd_capacity


def get_power_pp(nonlinear_class, chn):
    power1 = nonlinear_class.get_power_fixed_from_SNR(chn)
    return power1


def get_regime_pp(config, power, tanh_factor=None, multiplying_factor=1):
    # Complex Alphabet is on
    if config["complex"]:
        if np.isnan(power) or np.isinf(power):
            breakpoint()
        real_x, imag_x, real_y, imag_y = get_PP_complex_alphabet_x_y(
            config, power, tanh_factor
        )

        regime_class = get_regime_class(
            config=config,
            alphabet_x=real_x,
            alphabet_y=real_y,
            power=power,
            tanh_factor=tanh_factor,
            alphabet_x_imag=imag_x,
            alphabet_y_imag=imag_y,
            multiplying_factor=multiplying_factor,
        )
    else:
        alphabet_x, alphabet_y, max_x, max_y = get_alphabet_x_y(
            config, power, tanh_factor
        )

        regime_class = get_regime_class(
            config=config,
            alphabet_x=alphabet_x,
            alphabet_y=alphabet_y,
            power=power,
            tanh_factor=tanh_factor,
            multiplying_factor=multiplying_factor,
        )
    return regime_class


def get_linear_cap(config, power1):
    if config["regime"] == 3:
        linear_cap = (
            1
            / 2
            * np.log(1 + power1 / (config["sigma_11"] ** 2 + config["sigma_12"] ** 2))
        )
    else:
        linear_cap = 1 / 2 * np.log(1 + power1 / (config["sigma_12"] ** 2))
    if config["complex"]:
        linear_cap = 2 * linear_cap
    return linear_cap


def get_gaussian_cap(regime_class, power, config):
    (cap_g,) = (
        gaussian_capacity(regime_class, power, complex_alphabet=config["complex"]),
    )
    return cap_g


def get_capacity_learned(regime_class, power, config, save_location, change):
    (
        cap_learned,
        max_pdf_x,
        max_alphabet_x,
        opt_capacity,
    ) = gd_capacity(config, power, regime_class)

    title = str(config["change"]) + "=" + str(change)
    save_new = save_location + "/opt_" + title + ".png"
    plot_opt(opt_capacity, save_new, title)

    return cap_learned, max_pdf_x


def get_mmse_cap(regime_class, res):
    res["MMSE"] = bound_backtracing_check(
        res["MMSE"],
        reg_mmse_bound_numerical(regime_class),
    )
    return res


def plot_res_pp(res, change_range, save_location, config):
    markers = ["o", "s", "D", "v", "^", "p", "P", "*", "X", "d"]
    linestyle = ["--", "-.", ":", "-"]

    fig, ax = plt.subplots()
    ind = 0
    for key in res.keys():
        ax.plot(
            change_range,
            res[key],
            label=key,
            marker=markers[ind],
            linestyle=linestyle[ind],
        )
        ind += 1
    ax.legend(loc="best", fontsize=12)
    ax.set_xlabel("SNR", fontsize=12)
    ax.set_ylabel("Rate", fontsize=12)
    title = (
        "Saturation ="
        + str(config["Saturation_to_Noise"])
        + ", N1 = "
        + str(config["N_1"])
        + ", N2 = "
        + str(config["N_2"]),
    )
    if config["ADC"]:
        title = title + ", bits = " + str(config["b"])
    ax.set_title(title, fontsize=12)
    ax = grid_minor(ax)
    fig.savefig(save_location + "/rate.png")
    plt.close(fig)


def plot_pdf_pp(pdf, alph, save_location, config, change_range):

    for ind, pdf_x in enumerate(pdf):
        fig, ax = plt.subplots()
        ax.plot(alph[ind], pdf_x)
        ax.legend(loc="best", fontsize=12)
        ax.set_xlabel("Alphabet", fontsize=12)
        ax.set_ylabel("PDF", fontsize=12)
        title = (
            "Saturation ="
            + str(config["Saturation_to_Noise"])
            + ", N1 = "
            + str(config["N_1"])
            + ", N2 = "
            + str(config["N_2"])
        )

        if config["ADC"]:
            title = title + ", bits = " + str(config["b"])
        title = title + ", SNR = " + str(change_range[ind])
        ax.set_title(title, fontsize=12)
        ax = grid_minor(ax)
        fig.savefig(save_location + "/pdf" + str(change_range[ind]) + ".png")
        plt.close(fig)
