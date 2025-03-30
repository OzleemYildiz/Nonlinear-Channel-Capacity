import numpy as np
from gaussian_capacity import gaussian_interference_capacity
from gd import gradient_descent_on_interference, get_fixed_interferer
import matplotlib.pyplot as plt
import os
from utils import grid_minor, plot_opt
from utils_interference import get_int_regime

from nonlinearity_utils import get_derivative_of_nonlinear_fn


def get_save_location(config, pp=False):
    save_location = config["output_dir"] + "/"
    if config["complex"]:
        save_location = save_location + "Complex_"

    save_location = (
        save_location
        + config["cons_str"]
        + "_R="
        + str(config["regime"])
        + "_Sat="
        + str(config["Saturation_to_Noise"])
        + "_N1="
        + str(config["N_1"])
        + "_N2="
        + str(config["N_2"])
        + "_a="
        + str(config["int_ratio"])
    )

    if pp:
        save_location = (
            save_location
            + "_SNR="
            + str(config["snr_min"])
            + "-"
            + str(config["snr_min"] + config["snr_range"])
        )
    else:
        if config["change"] == "INR":
            save_location = save_location + "_SNR2="

        elif config["change"] == "SNR":
            save_location = save_location + "_SNR1="

        save_location = (
            save_location
            + str(config["snr_min"])
            + "-"
            + str(config["snr_min"] + config["snr_range"])
        )

        if config["change"] == "INR":
            save_location = save_location + "_SNR1="
        elif config["change"] == "SNR":
            save_location = save_location + "_SNR2="
        save_location = save_location + str(config["snr_fixed"])

    if config["ADC"]:
        save_location = save_location + "_ADC_b=" + str(config["bits"])
    save_location = save_location + "/"
    return save_location


def get_power(chn, nonlinear_class, config):
    if config["change"] == "INR":
        power2 = nonlinear_class.get_power_fixed_from_INR(chn)
        power1 = nonlinear_class.get_power_fixed_from_SNR(config["snr_fixed"])
    elif config["change"] == "SNR":
        power1 = nonlinear_class.get_power_fixed_from_SNR(chn)
        power2 = nonlinear_class.get_power_fixed_from_INR(config["snr_fixed"])

    # If I am running in Regime 1, I need to change the power2
    # Noise 1 is also Gaussian so I sum their powers to get the total power
    if config["regime"] == 1 and config["x2_type"] == 0:
        power2 = power2 + config["sigma_11"] ** 2

    return power1, power2


def get_linear_int_capacity(power_1, power_2, nonlinear_class, int_ratio, config):
    noise_power = nonlinear_class.get_total_noise_power()

    # Since in this condition, I am adding noise1 to the power2 (summation of Gaussian variances make a new Gaussian)
    if config["regime"] == 1 and config["x2_type"] == 0:
        int_power = (power_2 - config["sigma_11"] ** 2) * int_ratio**2
    else:
        int_power = power_2 * int_ratio**2

    snr_linear_ki = power_1 / noise_power

    snr_linear_tin = power_1 / (noise_power + int_power)

    linear_ki = 1 / 2 * np.log(1 + snr_linear_ki)
    linear_tin = 1 / 2 * np.log(1 + snr_linear_tin)

    if config["complex"]:  # 1/2 comes from real
        linear_ki = linear_ki * 2
        linear_tin = linear_tin * 2
    return linear_ki, linear_tin


def get_capacity_gaussian(regime_RX1, regime_RX2, pdf_x_RX2, int_ratio):

    # Which should be the case for our current scenarios
    if regime_RX1.config["x2_fixed"]:
        upd_RX2 = False
    else:
        raise ValueError("Not implemented yet")

    cap_g_tin, _ = gaussian_interference_capacity(
        reg_RX1=regime_RX1,
        reg_RX2=regime_RX2,
        int_ratio=int_ratio,
        tin_active=True,  # First, we apply tin
        pdf_x_RX2=pdf_x_RX2,
        upd_RX2=upd_RX2,
    )
    cap_g_ki, _ = gaussian_interference_capacity(
        reg_RX1=regime_RX1,
        reg_RX2=regime_RX2,
        int_ratio=int_ratio,
        tin_active=False,  # Then, we apply ki
        pdf_x_RX2=pdf_x_RX2,
        upd_RX2=upd_RX2,
        reg3_active=True,
    )
    return cap_g_ki, cap_g_tin


def get_capacity_learned(
    regime_RX1, regime_RX2, pdf_x_RX2, int_ratio, config, save_location, change
):
    if regime_RX1.config["x2_fixed"]:
        upd_RX2 = False
        # Since X2 is always fixed
        lambda_sweep = [1]
    else:
        raise ValueError("Not implemented yet")

    # TIN
    (
        max_sum_cap,
        pdf_learned_tin,
        max_pdf_x_RX2,
        cap_learned_tin,
        max_cap_RX2,
        save_opt_sum_capacity,
    ) = gradient_descent_on_interference(
        config=config,
        reg_RX1=regime_RX1,
        reg_RX2=regime_RX2,
        lambda_sweep=lambda_sweep,
        int_ratio=int_ratio,
        tin_active=True,  # First, we apply tin
        pdf_x_RX2=pdf_x_RX2,
        upd_RX2=upd_RX2,
    )
    # KI
    title = "TIN_" + str(config["change"]) + "=" + str(change)
    save_new = save_location + "/opt_" + title + ".png"

    plot_opt(save_opt_sum_capacity[0], save_new, title)

    (
        max_sum_cap2,
        pdf_learned_ki,
        max_pdf_x_RX2,
        cap_learned_ki,
        max_cap_RX2,
        save_opt_sum_capacity2,
    ) = gradient_descent_on_interference(
        config=config,
        reg_RX1=regime_RX1,
        reg_RX2=regime_RX2,
        lambda_sweep=lambda_sweep,
        int_ratio=int_ratio,
        tin_active=False,  # Then, we apply ki
        pdf_x_RX2=pdf_x_RX2,
        upd_RX2=upd_RX2,
        reg3_active=True,
    )

    title = "KI_" + str(config["change"]) + "=" + str(change)
    save_new = save_location + "/opt_" + title + ".png"

    plot_opt(save_opt_sum_capacity2[0], save_new, title)

    return cap_learned_ki, cap_learned_tin, pdf_learned_tin, pdf_learned_ki


def plot_int_res(res, config, save_location, change_range):

    markers = ["o", "s", "v", "D", "P", "X", "H", "d", "p", "x"]
    linestyle = ["solid", "dashed", "dashdot", "dotted"]
    fig_ki, ax_ki = plt.subplots()  # KI
    fig_tin, ax_tin = plt.subplots()  # TIN
    for ind, key in enumerate(res["KI"].keys()):
        ax_ki.plot(
            res["Change_Range"],
            res["KI"][key],
            label=key,
            marker=markers[ind],
            linestyle=linestyle[ind],
        )
    ax_ki.set_xlabel(config["change"], fontsize=12)
    ax_ki.set_ylabel("Rate", fontsize=12)
    ax_ki.legend(loc="best", fontsize=12)
    if config["change"] == "INR":
        ax_ki.set_title(
            "Known Interference with SNR = " + str(config["snr_fixed"]), fontsize=12
        )
    else:
        ax_ki.set_title(
            "Known Interference with INR = " + str(config["snr_fixed"]), fontsize=12
        )
    ax_ki = grid_minor(ax_ki)
    plt.tight_layout()
    fig_ki.savefig(save_location + "rate_ki.png", bbox_inches="tight")
    plt.close(fig_ki)

    for ind, key in enumerate(res["TIN"].keys()):
        ax_tin.plot(
            res["Change_Range"],
            res["TIN"][key],
            label=key,
            marker=markers[ind],
            linestyle=linestyle[ind],
        )
    ax_tin.set_xlabel(config["change"], fontsize=12)
    ax_tin.set_ylabel("Rate", fontsize=12)
    ax_tin.legend(loc="best", fontsize=12)
    if config["change"] == "INR":
        ax_tin.set_title(
            "Treat Interference as Noise with SNR = " + str(config["snr_fixed"]),
            fontsize=12,
        )
    else:
        ax_tin.set_title(
            "Treat Interference as Noise with INR = " + str(config["snr_fixed"]),
            fontsize=12,
        )
    ax_tin = grid_minor(ax_tin)
    plt.tight_layout()
    fig_tin.savefig(save_location + "rate_tin.png", bbox_inches="tight")
    plt.close(fig_tin)


def plot_int_pdf(pdf, config, save_location, change_range, alph):

    markers = ["o", "s", "v", "D", "P", "X", "H", "d", "p", "x"]
    linestyle = ["solid", "dashed", "dashdot", "dotted"]

    for ind_c, chn in enumerate(change_range):
        fig, ax = plt.subplots()
        ind_m = 0
        for key in pdf.keys():

            plt.plot(
                alph[ind_c].reshape(-1),
                pdf[key][ind_c].reshape(-1),
                label=key,
                marker=markers[ind_m],
                linestyle=linestyle[ind_m],
            )
            ind_m += 1

        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("PDF", fontsize=12)

        if config["change"] == "INR":
            ax.set_title(
                "INR = " + str(chn) + ", SNR = " + str(config["snr_fixed"]), fontsize=12
            )
        else:
            ax.set_title(
                "INR = " + str(config["snr_fixed"]) + ", SNR = " + str(chn), fontsize=12
            )
        ax.legend(loc="best", fontsize=12)
        ax = grid_minor(ax)
        plt.tight_layout()
        fig.savefig(
            save_location + "pdf_" + str(config["change"]) + "=" + str(chn) + ".png",
            bbox_inches="tight",
        )
        plt.close(fig)


def get_linear_app_int_capacity(regime_RX2, config, power1, pdf_x_RX2, int_ratio):
    # I also deleted multiplying_factor since the current file does not use gain
    # Removed TIN - we dont use this

    # The current regime RX2 involves the noise of RX1
    if config["regime"] == 1 and config["x2_type"] == 0:
        power2_upd = regime_RX2.power - config["sigma_11"] ** 2
        _, regime_RX2_upd = get_int_regime(
            config, power1, power2_upd, int_ratio, tanh_factor=0, tanh_factor2=0
        )
        pdf_x_RX2_upd = get_fixed_interferer(
            config,
            regime_RX2_upd,
            config["x2_type"],
        )
    else:

        pdf_x_RX2_upd = pdf_x_RX2
        regime_RX2_upd = regime_RX2

    deriv_func = get_derivative_of_nonlinear_fn(
        regime_RX2_upd.config, tanh_factor=regime_RX2_upd.tanh_factor
    )

    if config["complex"]:
        d_phi_s = (
            abs(
                deriv_func(int_ratio * abs(regime_RX2_upd.alphabet_x))
                * torch.exp(1j * torch.angle(regime_RX2_upd.alphabet_x))
            )
            ** 2
        )
    else:
        d_phi_s = abs(deriv_func(int_ratio * regime_RX2_upd.alphabet_x)) ** 2

    ki = np.log(
        1
        + (d_phi_s * power1)
        / (config["sigma_11"] ** 2 * d_phi_s + config["sigma_12"] ** 2)
    )

    approx_cap_ki = ki @ pdf_x_RX2_upd

    if not config["complex"]:
        approx_cap_ki = approx_cap_ki / 2

    return approx_cap_ki.detach().numpy()
