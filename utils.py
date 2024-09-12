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
    n, m = len(alphabet_x), 1
    if n == 0:
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

    objective = cp.Minimize(cp.pnorm(p - p_hat, p=2))
    problem = cp.Problem(objective, constraints)

    if cons_type == 1:
        cvxpylayer = CvxpyLayer(problem, parameters=[A, mu, p], variables=[p_hat])
        power = torch.tensor([power]).float()
        A_x = alphabet_x * alphabet_x
        A_x = A_x.reshape(1, -1)
        (solution,) = cvxpylayer(A_x, power, pdf_x)
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
):
    pdf_x = project_pdf(
        pdf_x,
        regime_class.config["cons_type"],
        regime_class.alphabet_x,
        regime_class.power,
    )
    # loss = -regime_class.capacity(pdf_x=pdf_x)
    # print("What we did", -loss)

    # Make sure that projection is working
    assert pdf_x is not None, "pdf_x is None"
    assert (
        torch.abs(torch.sum(pdf_x) - 1) <= 1e-5
    ), "pdf_x is not normalized the sum is " + str(torch.sum(pdf_x).detach().numpy())
    if torch.sum(pdf_x < 0) == 0:
        # , "pdf_x has negative values " + str(
        # pdf_x[np.where((pdf_x < 0))].detach().numpy())
        for ind, i in enumerate(pdf_x[np.where((pdf_x < 0))]):
            if pdf_x[ind] <= 1e-5:
                pdf_x[ind] = abs(pdf_x[ind])
            else:
                raise ValueError(
                    "pdf_x has negative values "
                    + str(pdf_x[np.where((pdf_x < 0))].detach().numpy())
                )

    cap = regime_class.capacity_like_ba(pdf_x)
    # print("What they did", cap)
    loss = -cap
    return loss


def plot_res(opt_capacity, pdf_x, alphabet_x, snr, cons_str, file_name, nonlinearity):
    plt.figure(figsize=(5, 4))
    plt.plot(opt_capacity)
    plt.xlabel("iteration")
    plt.ylabel("capacity")
    plt.savefig(
        file_name
        + "iteration_snr="
        + str(int(snr))
        + "_"
        + cons_str
        + "_channel="
        + str(nonlinearity)
        + ".png"
    )
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.bar(alphabet_x, pdf_x.detach().numpy())
    plt.xlabel("X")
    plt.ylabel("pdf")
    plt.savefig(
        file_name
        + "pdfx_snr="
        + str(int(snr))
        + "_"
        + cons_str
        + "_channel="
        + str(nonlinearity)
        + ".png"
    )
    plt.close()


def plot_snr(
    snr_range,
    res,
    config,
    save_location=None,
):

    plt.figure(figsize=(5, 4))
    leg_str = []
    for keys in res.keys():
        plt.plot(snr_range, res[keys])
        leg_str.append(keys)
    plt.legend(leg_str)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Capacity")
    plt.grid()
    if save_location == None:
        plt.savefig(
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
            + ".png"
        )
    else:
        plt.savefig(save_location + "/Comp" + ".png")
    plt.close()


def plot_pdf_snr(map_snr_pdf, snr_range, config, save_location=None):

    plt.figure(figsize=(5, 4))
    for snr in snr_range:
        pdf_x, alphabet_x = map_snr_pdf[str(snr)]
        if not isinstance(pdf_x, np.ndarray):
            pdf_x = pdf_x.detach().numpy()
        if not isinstance(alphabet_x, np.ndarray):
            alphabet_x = alphabet_x.detach().numpy()

        act_pdf_x = pdf_x > 0
        act_alp = alphabet_x[act_pdf_x]
        plt.scatter(
            snr * np.ones_like(act_alp),
            act_alp,
            s=pdf_x[act_pdf_x] * 100,
        )
    plt.xlabel("SNR (dB)")
    plt.ylabel("X")
    plt.grid()
    if save_location == None:
        plt.savefig(
            config["output_dir"]
            + "/"
            + config["cons_str"]
            + "_nonlinearity="
            + str(config["nonlinearity"])
            + "_regime="
            + str(config["regime"])
            + "_gd_"
            + str(config["gd_active"])
            + "/pdf_snr.png"
        )
    else:
        plt.savefig(save_location + "/pdf_snr.png")
    plt.close()


def generate_alphabet_x_y(config, power):
    nonlinear_func = return_nonlinear_fn(config)
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

    sample_num = math.ceil(2 * max_y / config["delta_y"]) + 1
    alphabet_y = torch.linspace(-max_y, max_y, sample_num)

    # Gaussian Result
    # alphabet_x
    sample_num_g = math.ceil(2 * max_x / config["delta_y"])
    alphabet_x = torch.linspace(-max_x, max_x, sample_num_g)

    return alphabet_x, alphabet_y, max_x, max_y


def return_regime_class(config, alphabet_x, alphabet_y, power):
    if config["regime"] == 1:
        regime_class = First_Regime(alphabet_x, alphabet_y, config, power)
    elif config["regime"] == 2:
        regime_class = Second_Regime(alphabet_x, config, power)
    elif config["regime"] == 3:
        regime_class = Third_Regime(alphabet_x, alphabet_y, config, power)
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


def read_config():
    # READ CONFIG FILE
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="arguments.yml",
        help="Configure of post processing",
    )
    args = parser.parse_args()
    config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    os.makedirs(config["output_dir"], exist_ok=True)
    # Constraint type of the system
    if config["cons_type"] == 1:
        config["cons_str"] = "Avg"
    elif config["cons_type"] == 0:
        config["cons_str"] = "Peak"
    else:
        config["cons_str"] = "First"
    return config
