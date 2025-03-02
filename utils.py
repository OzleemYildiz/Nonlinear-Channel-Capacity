import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp
import os
import random
from nonlinearity_utils import get_nonlinear_fn, get_derivative_of_nonlinear_fn
from First_Regime import First_Regime
from Second_Regime import Second_Regime
from Third_Regime import Third_Regime
import argparse
import yaml


def project_pdf(pdf_x, config, alphabet_x, power):
    # pdf cannot be negative
    # pdf_x = torch.relu(pdf_x)
    # sum of pdf is 1
    # pdf_x = pdf_x/torch.sum(pdf_x)
    # average power constraint

    cons_type = config["cons_type"]
    mul_factor = config["multiplying_factor"]

    if check_pdf_x_region(pdf_x, alphabet_x, cons_type, power, mul_factor):
        return pdf_x

    n, m = len(alphabet_x), 1
    if n == 0:
        print("While Projecting, alphabet_x is empty")
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
            print("Error in Projecting- Solution cannot be found")
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
            regime_class.config,
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
        print("Error in loss function - pdf_x or cap is nan or there is <0 in pdf_x")
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
            ax.legend(loc="best", fontsize=10)
            ax.set_xlabel(r"iteration", fontsize=10)
            ax.set_ylabel(r"capacity", fontsize=10)
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
            ax.set_title(title + ", lambda = " + str(format(lmbd, ".1f")), fontsize=10)
            fig.savefig(
                save_location
                + "iteration_lambda="
                + str(format(lmbd, ".1f"))
                + "_"
                + res_str
                + ".png",
                bbox_inches="tight",
            )

        plt.close()

        list_line = ["-", "--", "-.", ":"]
        scatter_style = ["o", "<", "x", "v"]  # ",", "^", "x", ">"]
        index = 0

        fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)

        # Check if any alphabet is complex
        complex_run = False
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
        ax.legend(loc="best", fontsize=10)
        ax.set_xlabel(r"X", fontsize=10)
        ax.set_ylabel(r"PDF", fontsize=10)
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
        ax.set_title(title + ", lambda = " + str(format(lmbd, ".1f")), fontsize=10)
        fig.savefig(
            save_location
            + "pdfx_lambda="
            + str(format(lmbd, ".1f"))
            + res_str
            + ".png",
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
        ind = np.mod(ind + 1, len(line_styles))
        leg_str.append(keys)

    if low_tarokh is not None and (config["regime"] == 1 or config["regime"] == 3):
        ax.plot(low_tarokh["SNR"], low_tarokh["Lower_Bound"], "--", linewidth=3)
        leg_str.append("Lower Bound Tarokh")

    leg_str_new = [ind.replace("_", " ") for ind in leg_str]

    ax.legend(leg_str_new, loc="best", fontsize=10)
    if not config["time_division_active"]:
        ax.set_xlabel(r"SNR (dB)", fontsize=10)
    else:
        ax.set_xlabel(r"Time Division Ratio", fontsize=10)

    ax.set_ylabel(r"Rate", fontsize=10)
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
        fontsize=10,
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
    map_pdf,
    range_change,
    config,
    save_location=None,
    file_name=None,
    map_opt=None,
    multiplying_factor=None,
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

    snr_change, noise_power = regime_dependent_snr(config)
    for ind, chn in enumerate(range_change):
        power = (10 ** (chn / 10)) * noise_power

        if config["power_change_active"] and config["time_division_active"]:
            pdf_x, alphabet_x = map_pdf["Chng" + str(int(chn)) + "ind=0"]
        else:
            if config["hardware_params_active"]:
                power = power * 10 ** multiplying_factor[ind]

            pdf_x, alphabet_x = map_pdf["Chng" + str(int(power * 10))]

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
        ax.set_ylabel(r"X", fontsize=10)
    elif config["time_division_active"] and not config["complex"]:
        ax.set_xlabel(r"Time Division Ratio", fontsize=10)
        ax.set_ylabel(r"X", fontsize=10)
    else:  # Complex Domain
        ax.legend(loc="best", fontsize=10)
        ax.set_xlabel(r"Re(X)", fontsize=10)
        ax.set_ylabel(r"Im(X)", fontsize=10)

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
        fontsize=10,
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

    for ind, chn in enumerate(range_change):
        power = (10 ** (chn / 10)) * noise_power
        if config["power_change_active"] and config["time_division_active"]:
            pdf_x, alphabet_x = map_pdf["Chng" + str(int(chn * 100)) + "ind=0"]
        else:
            if config["hardware_params_active"]:
                power = power * 10 ** multiplying_factor[ind]

            pdf_x, alphabet_x = map_pdf["Chng" + str(int(power * 10))]
        save_new = save_location + "/pdf_" + str(int(chn * 100)) + ".png"
        fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)
        if config["complex"]:
            alphabet_x_re = np.real(alphabet_x)
            alphabet_x_im = np.imag(alphabet_x)
            ax.scatter(alphabet_x_re, alphabet_x_im, s=pdf_x * 100)
            ax.set_xlabel(r"Re(X)", fontsize=10)
            ax.set_ylabel(r"Im(X)", fontsize=10)
        else:
            ax.bar(alphabet_x, pdf_x, linewidth=3)
            ax.set_xlabel(r"X", fontsize=10)
            ax.set_ylabel(r"PDF", fontsize=10)

        ax = grid_minor(ax)
        title = config["title"] + " SNR = " + str(round(chn, 2))
        ax.set_title(
            title,
            fontsize=10,
        )
        fig.savefig(save_new, bbox_inches="tight")
        plt.close()

        if map_opt is not None and len(map_opt.keys()) != 0:
            if config["power_change_active"] and config["time_division_active"]:
                opt_cap = map_opt["Chng" + str(int(chn * 100)) + "ind=0"]
            else:
                opt_cap = map_opt["Chng" + str(int(power * 10))]
            save_new = save_location + "/opt_" + str(int(chn * 100)) + ".png"
            plot_opt(opt_cap, save_new, title)


def plot_opt(opt_cap, save_new, title):
    fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)

    ax.plot(opt_cap)
    ax.set_xlabel("iteration", fontsize=10)
    ax.set_ylabel("Rate", fontsize=10)
    ax.set_title(title, fontsize=10)
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
    nonlinear_func = get_nonlinear_fn(config, tanh_factor)

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
        non = nonlinear_func(max_x)
    # phi(X+Z_1)
    elif config["regime"] == 2:
        max_y = nonlinear_func(max_x + config["sigma_1"] * config["stop_sd"])

    # phi(X+Z_1)+Z_2
    elif config["regime"] == 3:
        max_y = (
            nonlinear_func(max_x + config["sigma_1"] * config["stop_sd"])
            + config["sigma_2"] * config["stop_sd"]
        )
        non = nonlinear_func(max_x + config["sigma_1"] * config["stop_sd"])

    # Keep the number of samples fixed instead of delta

    delta = min(2 * max_x / min_samples, 2 * max_y / min_samples, 2 * non / min_samples)
    # delta = min(2 * max_x / min_samples, 2 * max_y / min_samples)

    # Check if Z1 will have enough samples - It gets included only for 3rd Regime
    # if config["regime"] == 3:
    #     max_z1 = config["stop_sd"] * config["sigma_1"] ** 2
    #     delta = min(delta, 2 * max_z1 / min_samples)

    # else:
    #     delta_y = config["delta_y"]

    max_y = max_y + (delta - (max_y % delta))
    max_x = max_x + (delta - (max_x % delta))
    return max_x, max_y, delta


def get_alphabet_x_y(config, power, tanh_factor, bound=False):
    max_x, max_y, delta = get_max_alphabet_PP(
        config, power, tanh_factor, config["min_samples"], bound
    )
    # Create the alphabet with the fixed delta
    try:
        alphabet_x = torch.arange(-max_x, max_x + delta / 2, delta)
        alphabet_y = torch.arange(-max_y, max_y + delta / 2, delta)
    except:
        print("Error in creating alphabet")
        breakpoint()

    if len(alphabet_x) == 0:
        print("Alphabet X is empty, while creating alphabet")
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
    multiplying_factor=1,
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
            multiplying_factor=multiplying_factor,
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
            multiplying_factor=multiplying_factor,
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
    # Maximum value that x should take
    nonlinear_func1 = get_nonlinear_fn(config, tanh_factor)
    nonlinear_func2 = get_nonlinear_fn(config, tanh_factor2)
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

    # Maximum value that y should take
    if config["regime"] == 1:
        # Note that both X1 and X2 are the same power
        max_y_1 = (
            nonlinear_func1(max_x + int_ratio * max_x2)
            + config["sigma_12"] * config["stop_sd"]
        )
        max_y_2 = nonlinear_func2(max_x2) + config["sigma_22"] * config["stop_sd"]
    elif config["regime"] == 3:
        max_y_1 = (
            nonlinear_func1(
                max_x + int_ratio * max_x2 + config["sigma_11"] * config["stop_sd"]
            )
            + config["sigma_12"] * config["stop_sd"]
        )
        max_y_2 = (
            nonlinear_func2(max_x2 + config["sigma_21"] * config["stop_sd"])
            + config["sigma_22"] * config["stop_sd"]
        )
    else:
        raise ValueError("Regime not defined")

    # The necessary separation between the points
    if int_ratio > 0 and int_ratio <= 1:
        delta = min(
            2 * max_x2 / config["min_samples"],
            2 * max_x / (config["min_samples"] * int_ratio),
            2 * max_y_1 / config["min_samples"],
            2 * max_y_2 / config["min_samples"],
        )

        # Regime 3 requires us to discretize Z11 as well
        if config["regime"] == 3:
            max_z11 = config["stop_sd"] * config["sigma_11"] ** 2
            max_z21 = config["stop_sd"] * config["sigma_21"] ** 2
            delta = min(
                delta,
                2 * max_z11 / config["min_samples"],
                2 * max_z21 / config["min_samples"],
            )

        delta_x2 = delta
        delta_x1 = int_ratio * delta

    elif int_ratio > 1:
        delta = min(
            2 * int_ratio * max_x2 / config["min_samples"],
            2 * max_x / config["min_samples"],
            2 * max_y_1 / config["min_samples"],
            2 * max_y_2 / config["min_samples"],
        )
        # Regime 3 requires us to discretize Z11 as well
        if config["regime"] == 3:
            max_z11 = config["stop_sd"] * config["sigma_11"] ** 2
            max_z21 = config["stop_sd"] * config["sigma_21"] ** 2
            delta = min(
                delta,
                2 * max_z11 / config["min_samples"],
                2 * max_z21 / config["min_samples"],
            )

        delta_x1 = delta
        delta_x2 = delta / int_ratio
    else:
        raise ValueError("Interference ratio must be positive")

    max_x_1 = max_x + (delta_x1 - (max_x % delta_x1))
    max_x_2 = max_x2 + (delta_x2 - (max_x2 % delta_x2))

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
            config,
            power1 / 2,
            power2 / 2,
            int_ratio,
            tanh_factor,
            tanh_factor2,
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
    ax.set_xlabel(r"Average Power", fontsize=10)
    ax.set_ylabel(r"Capacity", fontsize=10)
    ax.legend(leg_str, loc="best", fontsize=10)
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
    res_str="",
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

    ax.set_xlabel(r"Rate 1", fontsize=10)
    ax.set_ylabel(r"Rate 2", fontsize=10)

    if power2 is not None:
        ax.set_title(
            r"Power User 1 = "
            + str(int(power1))
            + " Power User 2 = "
            + str(int(power2)),
            fontsize=10,
        )
    else:
        ax.set_title(r"Power = " + str(int(power1)), fontsize=10)
    if res_str != "":
        title = res_str.replace("_", ", ")
        ax.set_title(title, fontsize=10)

    ax.legend(loc="best", fontsize=10)
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
        if res_str != "":
            fig.savefig(
                save_location + "/R1_R2_" + res_str + ".png",
                bbox_inches="tight",
            )
        else:
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
        print("Nan in Pdfs - Loss Interference")
        breakpoint()
    if upd_RX1:
        pdf_x_RX1 = project_pdf(
            pdf_x_RX1,
            reg_RX1.config,
            reg_RX1.alphabet_x,
            reg_RX1.power,
        )
    if upd_RX2:
        pdf_x_RX2 = project_pdf(
            pdf_x_RX2,
            reg_RX2.config,
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
        print("Nan in Capacity - Loss Interference")
        breakpoint()

    sum_capacity = lmbd * cap_RX1 + (1 - lmbd) * cap_RX2
    return -sum_capacity, cap_RX1, cap_RX2


def check_pdf_x_region(pdf_x, alphabet_x, cons_type, power, multiplying_factor=1):
    # We should check if the projection is necessary
    cond1 = torch.abs(torch.sum(pdf_x) - 1) < 1e-2  # sum of pdf is 1
    cond2 = torch.sum(pdf_x < 0) == 0  # pdf cannot be negative
    if cons_type == 1:
        cond3 = torch.sum(torch.abs(alphabet_x) ** 2 * pdf_x) <= power * (
            1 + 10 ** (-multiplying_factor - 2)
        )

    else:
        cond3 = True
    return cond1 and cond2 and cond3


def get_PP_complex_alphabet_x_y(config, power, tanh_factor, bound=False):
    if np.isnan(power) or power == np.inf:
        print("Power is nan or inf- complex alphabet creation")
        breakpoint()

    max_x, max_y, delta = get_max_alphabet_PP(
        config, power / 2, tanh_factor, config["min_samples"], bound
    )

    alphabet_x = torch.arange(-max_x, max_x + delta / 2, delta)
    delta = alphabet_x[1] - alphabet_x[0]
    # real_x, imag_x = torch.meshgrid([alphabet_x, alphabet_x])
    real_x = alphabet_x
    imag_x = alphabet_x.reshape(-1, 1)

    alphabet_y = torch.arange(-max_y, max_y + delta / 2, delta)

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
    multiplying_factor=1,
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
            multiplying_factor=multiplying_factor,
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
            multiplying_factor=multiplying_factor,
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

    ax.legend(loc="best", fontsize=10)
    ax.set_xlabel(str(config["change"]), fontsize=10)

    res_str_new = res_str.replace("_", ", ")

    ax.set_title(res_str_new, fontsize=10)
    ax.set_ylabel(r"R1", fontsize=10)
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


def grid_minor(ax):
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
    return ax


# pdf_y plotting
# If ADC is active, we need to plot quantized together
# Then save p_ys and p_y_given_x
def plot_pdf_y(regime_class, pdf_x, name_extra):
    mul_factor = regime_class.multiplying_factor
    pdf_x = pdf_x.detach()
    res_probs = {}
    res_probs["pdf_x"] = pdf_x
    res_probs["alph_x"] = regime_class.alphabet_x / 10 ** (mul_factor / 2)
    p_y = regime_class.pdf_y_given_x @ pdf_x
    alphabet_y = regime_class.alphabet_y / 10 ** (mul_factor / 2)
    res_probs["pdf_y"] = p_y
    res_probs["pdf_y_given_x"] = regime_class.pdf_y_given_x
    res_probs["alph_y"] = alphabet_y
    fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)
    if regime_class.config["complex"]:
        ax.scatter(
            alphabet_y.real,
            alphabet_y.imag,
            s=p_y * 100,
            label="PDF Y",
            color="c",
            marker="o",
        )
    else:
        ax.plot(alphabet_y, p_y, linewidth=3, label="PDF Y", color="c")
        ax.set_xlabel(r"Y", fontsize=10)
        ax.set_ylabel(r"PDF", fontsize=10)
        ax.set_ylim([0, 1.1 * torch.max(p_y)])
    ax = grid_minor(ax)
    lines, labels = ax.get_legend_handles_labels()

    if regime_class.config["ADC"]:

        q_pdf_y = regime_class.q_pdf_y_given_x @ pdf_x
        q_alph_y = regime_class.quant_locs / 10 ** (mul_factor / 2)
        res_probs["q_pdf_y"] = q_pdf_y
        res_probs["q_alph_y"] = q_alph_y
        if regime_class.config["complex"]:
            ax.scatter(
                q_alph_y.real,
                q_alph_y.imag,
                s=q_pdf_y * 100,
                label="Quantized PDF Y",
                color="r",
                marker="x",
            )
            ax.set_xlabel(r"Re(Y)", fontsize=10)
            ax.set_ylabel(r"Im(Y)", fontsize=10)
            lines, labels = ax.get_legend_handles_labels()
        else:
            ax2 = ax.twinx()
            ax2.bar(
                q_alph_y,
                q_pdf_y,
                label="Quantized PDF Y",
                color="r",
                width=10 ** np.round(np.log10(torch.min(abs(q_alph_y))) - 0.5),
            )
            ax2.set_ylabel("Quantized PDF Y", fontsize=10)
            ax2.set_ylim([0, 1.1 * torch.max(q_pdf_y)])
            lines2, labels2 = ax2.get_legend_handles_labels()
            lines = lines + lines2
            labels = labels + labels2
    ax.legend(lines, labels, loc=0)
    title = regime_class.config["title"]
    ax.set_title(
        title,
        fontsize=10,
    )

    fig.savefig(
        regime_class.config["save_location"] + "pdf_y_" + name_extra + ".png",
        bbox_inches="tight",
    )
    plt.close()

    # io.savemat(
    #     regime_class.config["save_location"] + "pdf_y_" + name_extra + ".mat",
    #     res_probs,
    # ) --> It was too big to save for Complex


# This kinda unnnecessary
def plot_quantized_x(regime_class, q_pdf, pdf_x, name_extra):
    fig, ax_left = plt.subplots(figsize=(5, 4), tight_layout=True)
    alphabet_x = regime_class.alphabet_x
    q_alph = regime_class.quant_locs

    ax_left.bar(q_alph, q_pdf, label="Quantized PDF", color="r")
    ax_left.set_ylabel("Quantized PDF", fontsize=10)
    ax_right = ax_left.twinx()
    ax_right.plot(alphabet_x, pdf_x, linewidth=3, label="Original PDF", color="c")
    ax_right.set_ylabel("Original PDF", fontsize=10)
    ax_left.set_xlabel(r"X", fontsize=10)

    lines, labels = ax_left.get_legend_handles_labels()
    lines2, labels2 = ax_right.get_legend_handles_labels()
    ax_left.legend(lines + lines2, labels + labels2, loc=0)

    # ax_right.legend(loc="best", fontsize=10)
    ax_left = grid_minor(ax_left)
    title = regime_class.config["title"]
    ax_left.set_title(
        title,
        fontsize=10,
    )
    fig.savefig(
        regime_class.config["save_location"] + "q_vs_orig_" + name_extra + ".png",
        bbox_inches="tight",
    )
    plt.close()


def plot_R1_R2_change(
    res_change, change_range, config, save_location, res_str, lambda_sweep
):

    markers = [
        "o",
        "v",
        "^",
        "<",
        ">",
        "x",
        "s",
        "p",
        "P",
        "*",
        "h",
        "H",
        "+",
        "X",
        "D",
        "d",
        "|",
        "_",
    ]
    ind_m = 0

    fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)
    for key in res_change["R1"].keys():
        if key == "Learned":
            for ind, l in enumerate(lambda_sweep):

                res_gather = [
                    res_change["R1"][key][c][ind] for c in range(len(change_range))
                ]
                ax.plot(
                    change_range,
                    res_gather,
                    label="R1 " + key + ", l=" + str(l),
                    linewidth=2,
                    linestyle="-.",
                    marker=markers[ind_m],
                )
                ind_m = np.mod(ind_m + 1, len(markers))

        else:
            ax.plot(
                change_range,
                res_change["R1"][key],
                label="R1 " + key,
                linewidth=2,
                linestyle=":",
                marker=markers[ind_m],
            )
            ind_m = np.mod(ind_m + 1, len(markers))
    for key in res_change["R2"].keys():
        if key == "Learned":
            for ind, l in enumerate(lambda_sweep):
                res_gather = [
                    res_change["R2"][key][c][ind] for c in range(len(change_range))
                ]
                ax.plot(
                    change_range,
                    res_gather,
                    label="R2 " + key + ", l=" + str(l),
                    linewidth=2,
                    linestyle="-.",
                    marker=markers[ind_m],
                )
                ind_m = np.mod(ind_m + 1, len(markers))
        ax.plot(
            change_range,
            res_change["R2"][key],
            label="R2 " + key,
            linewidth=2,
            linestyle="-",
            marker=markers[ind_m],
        )
        ind_m = np.mod(ind_m + 1, len(markers))

    ax.legend(loc="best", fontsize=8)
    ax.set_xlabel(str(config["change"]), fontsize=10)

    res_str_new = res_str.replace("_", ", ")

    ax.set_title(res_str_new, fontsize=10)
    ax.set_ylabel(r"R1", fontsize=10)
    ax = grid_minor(ax)

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
