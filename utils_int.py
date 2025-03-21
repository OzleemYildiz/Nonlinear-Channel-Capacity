import numpy as np
from gaussian_capacity import gaussian_interference_capacity
from gd import gradient_descent_on_interference
import matplotlib.pyplot as plt
import os
from utils import grid_minor


def get_save_location(config):
    save_location = config["output_dir"] + "/"
    if config["complex"]:
        save_location = save_location + "Complex_"

    save_location = (
        save_location
        + config["cons_str"]
        + "_Sat="
        + str(config["Saturation_to_Noise"])
        + "_N1="
        + str(config["N_1"])
        + "_N2="
        + str(config["N_2"])
    )

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
    return power1, power2


def get_linear_int_capacity(power_1, power_2, nonlinear_class, int_ratio, config):
    noise_power = nonlinear_class.get_total_noise_power()
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
    cap_g_tin, _ = gaussian_interference_capacity(
        reg_RX1=regime_RX1,
        reg_RX2=regime_RX2,
        int_ratio=int_ratio,
        tin_active=True,  # First, we apply tin
        pdf_x_RX2=pdf_x_RX2,
    )
    cap_g_ki, _ = gaussian_interference_capacity(
        reg_RX1=regime_RX1,
        reg_RX2=regime_RX2,
        int_ratio=int_ratio,
        tin_active=False,  # Then, we apply ki
        pdf_x_RX2=pdf_x_RX2,
    )
    return cap_g_ki, cap_g_tin


def get_capacity_learned(regime_RX1, regime_RX2, pdf_x_RX2, int_ratio, config):

    # Since X2 is always fixed
    lambda_sweep = [1]
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
    )
    # KI
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
    )
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

