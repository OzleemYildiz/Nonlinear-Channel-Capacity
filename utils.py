import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp
import os
import random
from nonlinearity_utils import return_nonlinear_fn, return_derivative_of_nonlinear_fn
from First_Regime import First_Regime
from Second_Regime import Second_Regime
from Third_Regime import Third_Regime
import argparse
import yaml


def project_pdf(pdf_x, cons_type, alphabet_x, power):
    # pdf cannot be negative
    # pdf_x = torch.relu(pdf_x)
    # sum of pdf is 1
    # pdf_x = pdf_x/torch.sum(pdf_x)
    # average power constraint
    if check_pdf_x_region(pdf_x, alphabet_x, cons_type, power):
        return pdf_x

    n, m = len(alphabet_x), 1
    if n == 0:
        breakpoint()
        raise ValueError("Alphabet_x is empty")
    p_hat = cp.Variable(n)
    p = cp.Parameter(n)
    A = cp.Parameter((m, n))
    mu = cp.Parameter(m)
    if cons_type == 1:
        # average power is in theconstraint
        constraints = [p_hat >= 1e-6, cp.sum(p_hat) == 1, A @ p_hat <= mu]
    elif cons_type == 0:  # peak power
        constraints = [p_hat >= 1e-6, cp.sum(p_hat) == 1]
    else:
        constraints = [p_hat >= 1e-6, cp.sum(p_hat) == 1, A @ p_hat <= mu]

    # objective = cp.Minimize(cp.pnorm(p - p_hat, p=2))
    objective = cp.Minimize(cp.sum_squares(p - p_hat))
    problem = cp.Problem(objective, constraints)

    if cons_type == 1:
        cvxpylayer = CvxpyLayer(problem, parameters=[A, mu, p], variables=[p_hat])
        power = torch.tensor([power]).float()
        A_x = abs(alphabet_x**2)
        A_x = A_x.reshape(1, -1)
        try:
            (solution,) = cvxpylayer(A_x, power, pdf_x)
        except:
            breakpoint()
    elif cons_type == 0:  # peak power
        cvxpylayer = CvxpyLayer(problem, parameters=[p], variables=[p_hat])
        (solution,) = cvxpylayer(pdf_x)
    else:  # first moment
        cvxpylayer = CvxpyLayer(problem, parameters=[A, mu, p], variables=[p_hat])
        power = torch.tensor([power]).float()
        A_x = torch.abs(alphabet_x)
        A_x = A_x.reshape(1, -1).to(torch.float32)
        # breakpoint()
        (solution,) = cvxpylayer(A_x, power, pdf_x.to(torch.float32))

        # if  torch.abs(alphabet_x)@solution > power:
        #   print('First moment constraint is not satisfied')
        #  breakpoint()
    # breakpoint()

    return solution


def loss(
    pdf_x,
    regime_class,
    project_active=True,
):

    if project_active:
        pdf_x = project_pdf(
            pdf_x,
            regime_class.config["cons_type"],
            regime_class.alphabet_x,
            regime_class.power,
        )
    if torch.sum(pdf_x < 0) > 0:
        pdf_x = torch.relu(pdf_x) + 1e-20

    cap = regime_class.new_capacity(pdf_x)
    # cap = regime_class.capacity_like_ba(pdf_x)
    # print("What they did", cap)
    loss = -cap

    if (
        len(pdf_x) == 0
        or torch.sum(pdf_x.isnan()) > 0
        or torch.sum(cap.isnan()) > 0
        or torch.sum(pdf_x < 0) > 0
    ):
        breakpoint()

    return loss


def plot_res(res_opt, res_pdf, res_alph, save_location, lmbd_sweep, res_str):
    # plt.rcParams["text.usetex"] = True
    title = res_str.replace("_", ", ")

    os.makedirs(save_location, exist_ok=True)
    for ind, lmbd in enumerate(lmbd_sweep):

        for key in res_opt.keys():
            fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)
            ax.plot(res_opt[key][ind], linewidth=3, label=key)
            ax.legend(loc="best", fontsize=12)
            ax.set_xlabel(r"iteration", fontsize=12)
            ax.set_ylabel(r"capacity", fontsize=12)
            ax.grid(
                visible=True,
                which="major",
                axis="both",
                color="lightgray",
                linestyle="-",
                linewidth=0.5,
            )
            plt.minorticks_on()
            ax.grid(
                visible=True,
                which="minor",
                axis="both",
                color="gainsboro",
                linestyle=":",
                linewidth=0.5,
            )
            ax.set_title(title + ", lambda = " + str(format(lmbd, ".1f")), fontsize=12)
            fig.savefig(
                save_location
                + "iteration_lambda="
                + str(format(lmbd, ".1f"))
                + "_"
                + key
                + ".png",
                bbox_inches="tight",
            )

        plt.close()

        list_line = ["-", "--", "-.", ":"]
        scatter_style = ["o", "<", "x", "v"]  # ",", "^", "x", ">"]
        index = 0

        fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)

        # Check if any alphabet is complex
        for key in res_pdf.keys():
            if res_alph[key].is_complex():
                complex_run = True
                break

        for key in res_pdf.keys():
            key_new = key.replace("_", " ")
            if complex_run:
                ax.scatter(
                    res_alph[key].real,
                    res_alph[key].imag,
                    s=res_pdf[key][ind] * 200,
                    label=key_new,
                    marker=scatter_style[index],
                )
            else:
                ax.plot(
                    res_alph[key],
                    res_pdf[key][ind],
                    label=key_new,
                    linestyle=list_line[index],
                    linewidth=3,
                )

            index = np.mod(index + 1, len(list_line))
        ax.legend(loc="best", fontsize=12)
        ax.set_xlabel(r"X", fontsize=12)
        ax.set_ylabel(r"PDF", fontsize=12)
        ax.grid(
            visible=True,
            which="major",
            axis="both",
            color="lightgray",
            linestyle="-",
            linewidth=0.5,
        )
        plt.minorticks_on()
        ax.grid(
            visible=True,
            which="minor",
            axis="both",
            color="gainsboro",
            linestyle=":",
            linewidth=0.5,
        )
        ax.set_title(title + ", lambda = " + str(format(lmbd, ".1f")), fontsize=12)
        fig.savefig(
            save_location + "pdfx_lambda=" + str(format(lmbd, ".1f")) + ".png",
            bbox_inches="tight",
        )
        plt.close()

        # ---- This was necessary for the sequential gradient descent version
        # plt.figure(figsize=(5, 4))
        # plt.plot(np.arange(len(max_sum_cap[ind])), max_sum_cap[ind])
        # plt.xlabel("iteration")
        # plt.ylabel("Sum capacity")
        # plt.savefig(
        #     save_location + "sum_capacity_lambda=" + str(format(lmbd, ".1f")) + ".png"
        # )
        # plt.close()


def plot_vs_change(
    change_range,
    res,
    config,
    save_location=None,
    low_tarokh=None,
):
    line_styles = ["-", "--", "-.", ":"]
    fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)
    leg_str = []
    ind = 0
    for keys in res.keys():
        ax.plot(change_range, res[keys], linewidth=3, linestyle=line_styles[ind])
        ind += 1
        leg_str.append(keys)

    if low_tarokh is not None and (config["regime"] == 1 or config["regime"] == 3):
        ax.plot(low_tarokh["SNR"], low_tarokh["Lower_Bound"], "--", linewidth=3)
        leg_str.append("Lower Bound Tarokh")

    leg_str_new = [ind.replace("_", " ") for ind in leg_str]

    ax.legend(leg_str_new, loc="best", fontsize=12)
    if not config["time_division_active"]:
        ax.set_xlabel(r"SNR (dB)", fontsize=12)
    else:
        ax.set_xlabel(r"Time Division Ratio", fontsize=12)

    ax.set_ylabel(r"Rate", fontsize=12)
    plt.minorticks_on()
    ax.grid(
        visible=True,
        which="major",
        axis="both",
        color="lightgray",
        linestyle="-",
        linewidth=0.5,
    )
    ax.grid(
        visible=True,
        which="minor",
        axis="both",
        color="gainsboro",
        linestyle=":",
        linewidth=0.5,
    )

    ax.set_title(
        config["title"],
        fontsize=12,
    )

    if save_location == None:
        fig.savefig(
            config["output_dir"]
            + "/"
            + config["cons_str"]
            + "_nonlinearity="
            + str(config["nonlinearity"])
            + "_regime="
            + str(config["regime"])
            + "_gd_"
            + str(config["gd_active"])
            + "/Comp"
            + ".png",
            bbox_inches="tight",
        )
    else:
        fig.savefig(
            save_location + "/Comp" + ".png",
            bbox_inches="tight",
        )
    plt.close()


def plot_pdf_vs_change(
    map_pdf, range_change, config, save_location=None, file_name=None, map_opt=None
):

    if file_name is None:
        if not config["time_division_active"]:
            file_name = "pdf_snr.png"
        else:
            file_name = "pdf_tau.png"

    fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)
    if config["time_division_active"] and not config["power_change_active"]:
        range_change = [range_change[0]]  # Because I only ran the code for one value

    markers = [
        "<",
        ",",
        "^",
        "o",
        "v",
        ">",
        "x",
        ".",
    ]
    if config["time_division_active"] and config["power_change_active"]:
        str_change = "alpha= "
    elif not config["time_division_active"]:
        str_change = "SNR= "
    else:
        return  # No need to plot anything because it's TDM without power change - pdf has already been plotted

    for ind, chn in enumerate(range_change):

        if config["power_change_active"]:
            pdf_x, alphabet_x = map_pdf["Chng" + str(int(chn * 100)) + "ind=0"]
        else:
            pdf_x, alphabet_x = map_pdf["Chng" + str(int(chn * 100))]

        if not isinstance(pdf_x, np.ndarray):
            pdf_x = pdf_x.detach().numpy()
        if not isinstance(alphabet_x, np.ndarray):
            alphabet_x = alphabet_x.detach().numpy()

        act_pdf_x = pdf_x > 0
        act_alp = alphabet_x[act_pdf_x]

        if not config["complex"]:  # Real Domain
            ax.scatter(
                chn * np.ones_like(act_alp),
                act_alp,
                s=pdf_x[act_pdf_x] * 100,
            )

        else:  # Complex Domain
            alphabet_x_re = np.real(act_alp)
            alphabet_x_im = np.imag(act_alp)
            ax.scatter(
                alphabet_x_re,
                alphabet_x_im,
                s=pdf_x[act_pdf_x] * 200,
                label=str_change + str(round(chn, 2)),
                marker=markers[ind % len(markers)],
            )

    if not config["time_division_active"] and not config["complex"]:
        ax.set_xlabel(r"SNR (dB)")
        ax.set_ylabel(r"X", fontsize=12)
    elif config["time_division_active"] and not config["complex"]:
        ax.set_xlabel(r"Time Division Ratio", fontsize=12)
        ax.set_ylabel(r"X", fontsize=12)
    else:  # Complex Domain
        ax.legend(loc="best", fontsize=12)
        ax.set_xlabel(r"Re(X)", fontsize=12)
        ax.set_ylabel(r"Im(X)", fontsize=12)

    ax.grid(
        visible=True,
        which="minor",
        axis="both",
        color="gainsboro",
        linestyle=":",
        linewidth=0.5,
    )
    plt.minorticks_on()
    ax.grid(
        visible=True,
        which="major",
        axis="both",
        color="lightgray",
        linestyle="-",
        linewidth=0.5,
    )

    ax.set_title(
        config["title"],
        fontsize=12,
    )
    if save_location == None:
        fig.savefig(
            config["output_dir"]
            + "/"
            + config["cons_str"]
            + "_nonlinearity="
            + str(config["nonlinearity"])
            + "_regime="
            + str(config["regime"])
            + "_gd_"
            + str(config["gd_active"])
            + "/"
            + file_name,
            bbox_inches="tight",
        )
    else:
        fig.savefig(save_location + "/" + file_name, bbox_inches="tight")
    plt.close()

    for chn in range_change:
        if config["power_change_active"]:
            pdf_x, alphabet_x = map_pdf["Chng" + str(int(chn * 100)) + "ind=0"]
        else:
            pdf_x, alphabet_x = map_pdf["Chng" + str(int(chn * 100))]
        save_new = save_location + "/pdf_" + str(int(chn * 100)) + ".png"
        fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)
        if config["complex"]:
            alphabet_x_re = np.real(alphabet_x)
            alphabet_x_im = np.imag(alphabet_x)
            ax.scatter(alphabet_x_re, alphabet_x_im, s=pdf_x * 100)
            ax.set_xlabel(r"Re(X)", fontsize=12)
            ax.set_ylabel(r"Im(X)", fontsize=12)
        else:
            ax.bar(alphabet_x, pdf_x, linewidth=3)
            ax.set_xlabel(r"X", fontsize=12)
            ax.set_ylabel(r"PDF", fontsize=12)
        ax.grid(
            visible=True,
            which="minor",
            axis="both",
            color="gainsboro",
            linestyle=":",
            linewidth=0.5,
        )
        ax.grid(
            visible=True,
            which="major",
            axis="both",
            color="lightgray",
            linestyle="-",
            linewidth=0.5,
        )
        plt.minorticks_on()
        title = config["title"] + " SNR = " + str(round(chn, 2))
        ax.set_title(
            title,
            fontsize=12,
        )
        fig.savefig(save_new, bbox_inches="tight")
        plt.close()

        if map_opt is not None and len(map_opt.keys()) != 0:
            if config["power_change_active"] and config["time_division_active"]:
                opt_cap = map_opt["Chng" + str(int(chn * 100)) + "ind=0"]
            else:
                opt_cap = map_opt["Chng" + str(int(chn * 100))]
            save_new = save_location + "/opt_" + str(int(chn * 100)) + ".png"
            plot_opt(opt_cap, save_new, title)


def plot_opt(opt_cap, save_new, title):
    fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)

    ax.plot(opt_cap)
    ax.set_xlabel("iteration", fontsize=12)
    ax.set_ylabel("Rate", fontsize=12)
    ax.set_title(title, fontsize=12)
    plt.minorticks_on()
    ax.grid(
        visible=True,
        which="minor",
        axis="both",
        color="gainsboro",
        linestyle=":",
        linewidth=0.5,
    )

    ax.grid(
        visible=True,
        which="major",
        axis="both",
        color="lightgray",
        linestyle="-",
        linewidth=0.5,
    )

    fig.savefig(save_new, bbox_inches="tight")
    plt.close()


def get_max_alphabet_PP(
    config,
    power,
    tanh_factor,
    min_samples,
    bound=False,
):
    nonlinear_func = return_nonlinear_fn(config, tanh_factor)
    if config["cons_type"] == 0:  # peak power
        peak_power = power
        # snr = peak_power/std
        max_x = np.sqrt(peak_power)
    elif config["cons_type"] == 1:  # average power
        avg_power = power
        # snr = avg_power/std
        # This could be changed
        # DOUBLE CHECK
        max_x = config["stop_sd"] * np.sqrt(avg_power)
        # max_x = stop_s*avg_power

        # If it's clipped after this value, it does not matter to put values outside
        # if config["nonlinearity"] == 5 and not bound:
        #     max_x = config["clipping_limit_x"]

    else:  # first moment
        first_moment = power  # E[|X|] < P
        max_x = config["stop_sd"] * first_moment

    # Note that all nonlinearity functions are one-one functions and non-decreasing
    # # phi(X)+ Z_2
    if config["regime"] == 1:
        # 0:linear
        max_y = nonlinear_func(max_x) + config["sigma_2"] * config["stop_sd"]
    # phi(X+Z_1)
    elif config["regime"] == 2:
        max_y = nonlinear_func(max_x + config["sigma_1"] * config["stop_sd"])
    # phi(X+Z_1)+Z_2
    elif config["regime"] == 3:
        max_y = (
            nonlinear_func(max_x + config["sigma_1"] * config["stop_sd"])
            + config["sigma_2"] * config["stop_sd"]
        )

    # Keep the number of samples fixed instead of delta
    delta_y = 2 * max_x / min_samples
    # if delta_y > config["delta_y"]:
    #     delta_y = config["delta_y"]

    # Check if Z1 will have enough samples - It gets included only for 3rd Regime
    if config["regime"] == 3:
        max_z1 = config["stop_sd"] * config["sigma_1"] ** 2
        sample_num = math.ceil(2 * max_z1 / delta_y) + 1
        if sample_num < min_samples:
            delta_y = 2 * max_z1 / min_samples

    # else:
    #     delta_y = config["delta_y"]

    max_y = max_y + (delta_y - (max_y % delta_y))
    max_x = max_x + (delta_y - (max_x % delta_y))

    return max_x, max_y, delta_y


def get_alphabet_x_y(config, power, tanh_factor, bound=False):
    max_x, max_y, delta_y = get_max_alphabet_PP(
        config, power, tanh_factor, config["min_samples"], bound
    )
    # Create the alphabet with the fixed delta
    alphabet_x = torch.arange(-max_x, max_x + delta_y / 2, delta_y)
    alphabet_y = torch.arange(-max_y, max_y + delta_y / 2, delta_y)

    if len(alphabet_x) == 0:
        breakpoint()
    return alphabet_x, alphabet_y, max_x, max_y


def get_regime_class(
    config,
    alphabet_x,
    alphabet_y,
    power,
    tanh_factor,
    alphabet_x_imag=0,
    alphabet_y_imag=0,
):
    if config["regime"] == 1:
        regime_class = First_Regime(
            alphabet_x=alphabet_x,
            alphabet_y=alphabet_y,
            config=config,
            power=power,
            tanh_factor=tanh_factor,
            sigma_2=config["sigma_2"],
            alphabet_x_imag=alphabet_x_imag,
            alphabet_y_imag=alphabet_y_imag,
        )
    elif config["regime"] == 2:
        regime_class = Second_Regime(alphabet_x, config, power)
    elif config["regime"] == 3:
        regime_class = Third_Regime(
            alphabet_x,
            alphabet_y,
            config,
            power,
            tanh_factor,
            sigma_1=config["sigma_1"],
            sigma_2=config["sigma_2"],
            alphabet_x_imag=alphabet_x_imag,
            alphabet_y_imag=alphabet_y_imag,
        )
    else:
        raise ValueError("Regime not defined")
    return regime_class


def regime_dependent_snr(config):
    # Number of SNR points to be evaluated
    if config["regime"] == 1:
        noise_power = config["sigma_2"] ** 2
    elif config["regime"] == 2:
        noise_power = config["sigma_1"] ** 2
    elif config["regime"] == 3:
        noise_power = config["sigma_1"] ** 2 + config["sigma_2"] ** 2
    snr_change = np.linspace(
        10 * np.log10(config["min_power_cons"] / (noise_power)),
        10 * np.log10(config["max_power_cons"] / (noise_power)),
        config["n_snr"],
    )
    return snr_change, noise_power


def read_config(args_name="arguments.yml"):
    # READ CONFIG FILE
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=args_name,
        help="Configure of post processing",
    )
    args = parser.parse_args()
    config = yaml.load(open("args/" + args.config, "r"), Loader=yaml.Loader)
    # os.makedirs(config["output_dir"], exist_ok=True)
    # Constraint type of the system
    if config["cons_type"] == 1:
        config["cons_str"] = "Avg"
    elif config["cons_type"] == 0:
        config["cons_str"] = "Peak"
    else:
        config["cons_str"] = "First"

    title = (
        "Regime="
        + str(config["regime"])
        + " phi="
        + str(config["nonlinearity"])
        + " N="
        + str(config["min_samples"])
    )

    config["title"] = title

    return config


def interference_dependent_snr(config, power):
    if config["regime"] == 1:
        snr1 = 10 * np.log10(power / (config["sigma_12"] ** 2))
        snr2 = 10 * np.log10(power / (config["sigma_22"] ** 2))
        inr1 = 10 * np.log10(
            power * config["int_ratio"] ** 2 / (config["sigma_12"] ** 2)
        )

    return snr1, snr2, inr1


def get_max_alphabet_interference(
    config, power1, power2, int_ratio, tanh_factor, tanh_factor2
):
    nonlinear_func1 = return_nonlinear_fn(config, tanh_factor)
    nonlinear_func2 = return_nonlinear_fn(config, tanh_factor2)
    if config["cons_type"] == 0:  # peak power
        peak_power = power1
        max_x = np.sqrt(peak_power)
        max_x2 = np.sqrt(power2)
    elif config["cons_type"] == 1:  # average power
        avg_power = power1
        max_x = config["stop_sd"] * np.sqrt(avg_power)
        max_x2 = config["stop_sd"] * np.sqrt(power2)
        # If it's clipped after this value, it does not matter to put values outside
        if config["nonlinearity"] == 5:
            max_x = config["clipping_limit_x"]
            max_x2 = config["clipping_limit_x"]
    else:  # first moment
        first_moment = power1  # E[|X|] < P
        max_x = config["stop_sd"] * first_moment
        max_x2 = config["stop_sd"] * power2

    if int_ratio > 0 and int_ratio <= 1:
        delta_y = 2 * max_x2 / config["min_samples"]  # !!! Changed this
        # if delta_y > config["delta_y"]:
        #     delta_y = config["delta_y"]
        delta_x2 = delta_y
        delta_x1 = int_ratio * delta_x2
    elif int_ratio > 1:
        delta_y = 2 * max_x / config["min_samples"]  # !!! Changed this
        # if delta_y > config["delta_y"]:
        #     delta_y = config["delta_y"]
        delta_x1 = delta_y
        delta_x2 = delta_x1 / int_ratio
    else:
        raise ValueError("Interference ratio must be positive")

    max_x_1 = max_x + (delta_x1 - (max_x % delta_x1))
    max_x_2 = max_x2 + (delta_x2 - (max_x2 % delta_x2))

    if config["regime"] == 1:
        # Note that both X1 and X2 are the same power
        max_y_1 = (
            nonlinear_func1(max_x_1 + int_ratio * max_x_2)
            + config["sigma_12"] * config["stop_sd"]
        )
        max_y_2 = nonlinear_func2(max_x_2) + config["sigma_22"] * config["stop_sd"]
    elif config["regime"] == 3:
        max_y_1 = (
            nonlinear_func1(
                max_x_1 + int_ratio * max_x_2 + config["sigma_11"] * config["stop_sd"]
            )
            + config["sigma_12"] * config["stop_sd"]
        )
        max_y_2 = (
            nonlinear_func2(max_x_2 + config["sigma_21"] * config["stop_sd"])
            + config["sigma_22"] * config["stop_sd"]
        )
    else:
        raise ValueError("Regime not defined")

    max_y_1 = max_y_1 + (delta_x1 - (max_y_1 % delta_x1))
    max_y_2 = max_y_2 + (delta_x2 - (max_y_2 % delta_x2))

    return max_x_1, max_y_1, delta_x1, max_x_2, max_y_2, delta_x2


def get_interference_alphabet_x_y(
    config, power1, power2, int_ratio, tanh_factor, tanh_factor2
):
    max_x_1, max_y_1, delta_x1, max_x_2, max_y_2, delta_x2 = (
        get_max_alphabet_interference(
            config, power1, power2, int_ratio, tanh_factor, tanh_factor2
        )
    )
    # Create the alphabet with the fixed delta
    alphabet_x_1 = torch.arange(-max_x_1, max_x_1 + delta_x1 / 2, delta_x1)
    alphabet_x_2 = torch.arange(-max_x_2, max_x_2 + delta_x2 / 2, delta_x2)

    alphabet_y_1 = torch.arange(-max_y_1, max_y_1 + delta_x1 / 2, delta_x1)
    alphabet_y_2 = torch.arange(-max_y_2, max_y_2 + delta_x2 / 2, delta_x2)
    return alphabet_x_1, alphabet_y_1, alphabet_x_2, alphabet_y_2


def get_interference_alphabet_x_y_complex(
    config, power1, power2, int_ratio, tanh_factor, tanh_factor2
):
    max_x_1, max_y_1, delta_x1, max_x_2, max_y_2, delta_x2 = (
        get_max_alphabet_interference(
            config, power1, power2, int_ratio, tanh_factor, tanh_factor2
        )
    )
    # Create the alphabet with the fixed delta
    alphabet_x_1 = torch.arange(-max_x_1, max_x_1 + delta_x1 / 2, delta_x1)
    real_x1 = alphabet_x_1.reshape(1, -1)
    imag_x1 = alphabet_x_1.reshape(-1, 1)
    alphabet_x_2 = torch.arange(-max_x_2, max_x_2 + delta_x2 / 2, delta_x2)
    real_x2 = alphabet_x_2.reshape(1, -1)
    imag_x2 = alphabet_x_2.reshape(-1, 1)

    alphabet_y_1 = torch.arange(-max_y_1, max_y_1 + delta_x1 / 2, delta_x1)
    real_y1 = alphabet_y_1.reshape(1, -1)
    imag_y1 = alphabet_y_1.reshape(-1, 1)

    alphabet_y_2 = torch.arange(-max_y_2, max_y_2 + delta_x2 / 2, delta_x2)
    real_y2 = alphabet_y_2.reshape(1, -1)
    imag_y2 = alphabet_y_2.reshape(-1, 1)
    return real_x1, imag_x1, real_y1, imag_y1, real_x2, imag_x2, real_y2, imag_y2


def plot_interference(res, config, save_location):
    power_range = np.logspace(
        np.log10(config["min_power_cons"]),
        np.log10(config["max_power_cons"]),
        config["n_snr"],
    )
    leg_str = []
    fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)
    for keys in res.keys():
        ax.plot(power_range, res[keys], linewidth=3)
        leg_str.append(keys)
    ax.set_xlabel(r"Average Power", fontsize=12)
    ax.set_ylabel(r"Capacity", fontsize=12)
    ax.legend(leg_str, loc="best", fontsize=12)
    ax.grid(
        visible=True,
        which="major",
        axis="both",
        color="lightgray",
        linestyle="-",
        linewidth=0.5,
    )
    plt.minorticks_on()
    ax.grid(
        visible=True,
        which="minor",
        axis="both",
        color="gainsboro",
        linestyle=":",
        linewidth=0.5,
    )

    fig.savefig(save_location + "/Comp" + ".png", bbox_inches="tight")
    plt.close()


# TODO: Save the image of the pdfs in the folder
# Be careful with x-range and y-range to be fixed so that the pdfs are comparable
def save_image_of_pdfs():
    pass


# TODO: Make gifs of the pdfs saved as images
def make_gifs_of_pdfs():
    pass


def plot_R1_R2_curve(
    res,
    power1,
    power2,
    save_location,
    config,
    res_gaus=None,
    res_tdm=None,
):
    markers = [".", ",", "o", "v", "^", "<", ">", "x"]
    ind = 0
    fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)
    # See how many plot we need to do
    hold_keys = []
    for keys in res["R1"].keys():
        hold_keys.append(keys)

    for keys in hold_keys:
        keys_new = keys.replace("_", " ")
        ax.scatter(
            res["R1"][keys], res["R2"][keys], label=keys_new, marker=markers[ind]
        )
        ind = np.mod(ind + 1, len(markers))

    # if res_gaus is not None:
    #     for keys in res_gaus.keys():
    #         keys_new = keys.replace("_", " ")
    #         ax.scatter(
    #             res_gaus[keys][0], res_gaus[keys][1], label=keys, marker=markers[ind]
    #         )
    #         ind = np.mod(ind + 1, len(markers))

    if res_tdm is not None:
        ax.plot(res_tdm["R1"], res_tdm["R2"], label="TDM", linewidth=3)

    ax.set_xlabel(r"Rate 1", fontsize=12)
    ax.set_ylabel(r"Rate 2", fontsize=12)

    if power2 is not None:
        ax.set_title(
            r"Power User 1 = "
            + str(int(power1))
            + " Power User 2 = "
            + str(int(power2)),
            fontsize=12,
        )
    else:
        ax.set_title(r"Power = " + str(int(power1)), fontsize=12)

    ax.legend(loc="best", fontsize=12)
    ax.grid(
        visible=True,
        which="major",
        axis="both",
        color="lightgray",
        linestyle="-",
        linewidth=0.5,
    )
    plt.minorticks_on()
    ax.grid(
        visible=True,
        which="minor",
        axis="both",
        color="gainsboro",
        linestyle=":",
        linewidth=0.5,
    )

    if power2 is not None:
        fig.savefig(
            save_location
            + "/R1_R2_pow1="
            + str(int(power1))
            + "_pow2="
            + str(int(power2))
            + ".png",
            bbox_inches="tight",
        )
    else:
        fig.savefig(
            save_location + "/R1_R2_pow=" + str(int(power1)) + ".png",
            bbox_inches="tight",
        )
    plt.close()


def loss_interference(
    pdf_x_RX1,
    pdf_x_RX2,
    reg_RX1,
    reg_RX2,
    int_ratio,
    tin_active,
    lmbd=0.5,
    upd_RX1=True,
    upd_RX2=True,
):
    # Interference loss function for GD
    if torch.sum(pdf_x_RX1.isnan()) > 0 or torch.sum(pdf_x_RX2.isnan()) > 0:
        breakpoint()
    if upd_RX1:
        pdf_x_RX1 = project_pdf(
            pdf_x_RX1,
            reg_RX1.config["cons_type"],
            reg_RX1.alphabet_x,
            reg_RX1.power,
        )
    if upd_RX2:
        pdf_x_RX2 = project_pdf(
            pdf_x_RX2,
            reg_RX2.config["cons_type"],
            reg_RX2.alphabet_x,
            reg_RX2.power,
        )

    # FIXME: Better write
    if torch.sum(pdf_x_RX1 < 0) > 0:
        # pdf_x_RX1 = torch.abs(pdf_x_RX1)
        # pdf_x_RX1 = torch.relu(pdf_x_RX1) + 1e-20
        pdf_x_RX1 = pdf_x_RX1 - torch.min(pdf_x_RX1[pdf_x_RX1 < 0]) + 1e-20
    if torch.sum(pdf_x_RX2 < 0) > 0:
        # Just RELU and make it positive
        # pdf_x_RX2 = torch.abs(pdf_x_RX2)
        # pdf_x_RX2 = torch.relu(pdf_x_RX2)+ 1e-20
        pdf_x_RX2 = pdf_x_RX2 - torch.min(pdf_x_RX2[pdf_x_RX2 < 0]) + 1e-20
        # breakpoint()

    if (tin_active and reg_RX1.config["x2_fixed"] == True) or reg_RX1.config[
        "x2_fixed"
    ] == False:
        cap_RX1 = reg_RX1.capacity_with_interference(
            pdf_x_RX1, pdf_x_RX2, reg_RX2.alphabet_x, int_ratio
        )

    elif not tin_active:  # Known interference
        cap_RX1 = reg_RX1.capacity_with_known_interference(
            pdf_x_RX1, pdf_x_RX2, reg_RX2.alphabet_x, int_ratio
        )
    cap_RX2 = reg_RX2.new_capacity(pdf_x_RX2)

    if torch.isnan(cap_RX1) or torch.isnan(cap_RX2):
        breakpoint()

    sum_capacity = lmbd * cap_RX1 + (1 - lmbd) * cap_RX2
    return -sum_capacity, cap_RX1, cap_RX2


def check_pdf_x_region(pdf_x, alphabet_x, cons_type, power):
    # We should check if the projection is necessary
    cond1 = torch.abs(torch.sum(pdf_x) - 1) < 1e-2  # sum of pdf is 1
    cond2 = torch.sum(pdf_x < 0) == 0  # pdf cannot be negative
    if cons_type == 1:
        cond3 = torch.sum(torch.abs(alphabet_x) ** 2 * pdf_x) <= power + 1e-3
    else:
        cond3 = True

    return cond1 and cond2 and cond3


def get_PP_complex_alphabet_x_y(config, power, tanh_factor, bound=False):
    max_x, max_y, delta_y = get_max_alphabet_PP(
        config, power, tanh_factor, config["min_samples"], bound
    )
    alphabet_x = torch.linspace(-max_x, max_x, config["min_samples"])
    delta = alphabet_x[1] - alphabet_x[0]
    # real_x, imag_x = torch.meshgrid([alphabet_x, alphabet_x])
    real_x = alphabet_x
    imag_x = alphabet_x.reshape(-1, 1)

    # FIXME: Not sure if avoiding zero is a good idea -- Check this
    alphabet_y = torch.arange(-max_y, max_y, delta)
    # real_y, imag_y = torch.meshgrid([alphabet_y, alphabet_y])
    real_y = alphabet_y
    imag_y = alphabet_y.reshape(-1, 1)

    return real_x, imag_x, real_y, imag_y


def get_regime_class_interference(
    alphabet_x_RX1,
    alphabet_x_RX2,
    alphabet_y_RX1,
    alphabet_y_RX2,
    config,
    power1,
    power2,
    tanh_factor,
    tanh_factor2,
    alphabet_x_RX1_imag=0,
    alphabet_x_RX2_imag=0,
    alphabet_y_RX1_imag=0,
    alphabet_y_RX2_imag=0,
):

    if config["regime"] == 1:
        # config["sigma_2"] = config["sigma_22"]

        f_reg_RX2 = First_Regime(
            alphabet_x=alphabet_x_RX2,
            alphabet_y=alphabet_y_RX2,
            config=config,
            power=power2,
            tanh_factor=tanh_factor2,
            sigma_2=config["sigma_22"],
            alphabet_x_imag=alphabet_x_RX2_imag,
            alphabet_y_imag=alphabet_y_RX2_imag,
        )
        # config["sigma_2"] = config["sigma_12"]
        f_reg_RX1 = First_Regime(
            alphabet_x=alphabet_x_RX1,
            alphabet_y=alphabet_y_RX1,
            config=config,
            power=power1,
            tanh_factor=tanh_factor,
            sigma_2=config["sigma_12"],
            alphabet_x_imag=alphabet_x_RX1_imag,
            alphabet_y_imag=alphabet_y_RX1_imag,
        )
        return f_reg_RX1, f_reg_RX2
    elif config["regime"] == 3:

        t_reg_RX2 = Third_Regime(
            alphabet_x=alphabet_x_RX2,
            alphabet_y=alphabet_y_RX2,
            config=config,
            power=power2,
            tanh_factor=tanh_factor2,
            sigma_1=config["sigma_21"],
            sigma_2=config["sigma_22"],
            alphabet_x_imag=alphabet_x_RX2_imag,
            alphabet_y_imag=alphabet_y_RX2_imag,
        )
        t_reg_RX1 = Third_Regime(
            alphabet_x=alphabet_x_RX1,
            alphabet_y=alphabet_y_RX1,
            config=config,
            power=power1,
            tanh_factor=tanh_factor,
            sigma_1=config["sigma_11"],
            sigma_2=config["sigma_12"],
            alphabet_x_imag=alphabet_x_RX1_imag,
            alphabet_y_imag=alphabet_y_RX1_imag,
        )
        return t_reg_RX1, t_reg_RX2
    else:
        raise ValueError("Regime not defined")


def plot_R1_vs_change(res_change, change_range, config, save_location, res_str):
    list_line = ["-", "--", "-.", ":"]

    fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)
    index = 0

    for keys in res_change.keys():
        keys_new = keys.replace("_", " ")
        ax.plot(
            change_range,
            res_change[keys],
            label=keys_new,
            linestyle=list_line[index],
            linewidth=3,
        )
        index = np.mod(index + 1, 4)

    ax.legend(loc="best", fontsize=12)
    ax.set_xlabel(str(config["change"]), fontsize=12)

    res_str_new = res_str.replace("_", ", ")

    ax.set_title(res_str_new, fontsize=12)
    ax.set_ylabel(r"R1", fontsize=12)
    plt.minorticks_on()
    ax.grid(
        visible=True,
        which="major",
        axis="both",
        color="lightgray",
        linestyle="-",
        linewidth=0.5,
    )
    ax.grid(
        visible=True,
        which="minor",
        axis="both",
        color="gainsboro",
        linestyle=":",
        linewidth=0.5,
    )

    fig.savefig(
        save_location + "/Comp_" + str(config["change"]) + "_" + res_str + ".png",
        bbox_inches="tight",
    )
    print(
        "Saved in ",
        str(config["change"]) + "_" + res_str + ".png",
    )
    plt.close()


# def loss_complex(pdf_x, regime_class, project_active=True):
#     if project_active:
#         breakpoint()
#         # FIXME : Gotta check projection

#         pdf_x = project_pdf(
#             pdf_x,
#             regime_class.config["cons_type"],
#             alphabet_x,
#             regime_class.power,
#         )
#     if torch.sum(pdf_x < 0) > 0:
#         pdf_x = torch.relu(pdf_x) + 1e-20

#     cap = regime_class.capacity_complex_PP(pdf_x)
#     loss = -cap
#     return loss


# plotting 2D - might be a good idea to plot 3D - future complex domain #TODO
# fig = plt.figure(figsize=(14,6))
# ax = fig.add_subplot(1, 1, 1, projection='3d')
# ax.plot_surface(regime_class.alphabet_x_re, regime_class.alphabet_x_im, pdf_x)
