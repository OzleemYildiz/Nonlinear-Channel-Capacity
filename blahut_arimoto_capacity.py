#!/usr/bin/env python
# coding: utf-8

# # Blauth-Arimotho Algorithm
# Assuming X and Y as input and output variables of the channel respectively and r(x) is the input distributions. <br>
# The capacity of a channel is defined by <br>
# $C = \max_{r(x)} I(X;Y) = \max_{r(x)} \sum_{x} \sum_{y} r(x) p(y|x) \log \frac{r(x) p(y|x)}{r(x) \sum_{\tilde{x}} r(\tilde{x})p(y|\tilde{x})}$


import numpy as np
from utils import (
    project_pdf,
    read_config,
    get_alphabet_x_y,
    get_regime_class,
    regime_dependent_snr,
    plot_pdf_vs_change,
    plot_vs_change,
    get_interference_alphabet_x_y,
    plot_R1_R2_curve,
)
import torch
import matplotlib.pyplot as plt
import os
from scipy import io
from First_Regime import First_Regime


# constraints are probability sums to 1 and non-negative
def blahut_arimoto(
    p_y_x: np.ndarray,
    alphabet_x: np.ndarray,
    average_power=True,
    log_base: float = 2,
    thresh: float = 1e-12,
    max_iter: int = 1e3,
    lambda_2: float = 0.1,
) -> tuple:
    """
    Maximize the capacity between I(X;Y)
    p_y_x: each row represnets probability assinmnet
    log_base: the base of the log when calaculating the capacity
    thresh: the threshold of the update, finish the calculation when gettting to it.
    max_iter: the maximum iterations of the calculation
    """

    # Input test
    assert np.abs(p_y_x.sum(axis=1).mean() - 1) < 1e-6
    assert p_y_x.shape[0] > 1

    # The number of inputs: size of |X|
    m = p_y_x.shape[0]

    # The number of outputs: size of |Y|
    n = p_y_x.shape[1]

    # Initialize the prior uniformly
    r = np.ones((1, m)) / m

    # Compute the r(x) that maximizes the capacity
    for iteration in range(int(max_iter)):

        q = r.reshape(-1, 1) * p_y_x
        q = q / np.sum(q, axis=0)

        if average_power:
            r1 = np.prod(np.power(q, p_y_x), axis=1) * np.exp(
                -lambda_2 * (alphabet_x**2)
            )
        else:  # Peak Power
            r1 = np.prod(np.power(q, p_y_x), axis=1)
        r1 = r1 / np.sum(r1)

        tolerance = np.linalg.norm(r1 - r)
        r = r1
        if tolerance < thresh:
            break

    # Calculate the capacity
    r = r.flatten()
    c = 0
    for i in range(m):
        if r[i] > 0:
            c += np.sum(r[i] * p_y_x[i, :] * np.log(q[i, :] / r[i] + 1e-16))
    c = c / np.log(log_base)
    return c, r


# constraints are probability sums to 1 and non-negative
def blahut_arimoto_torch(
    p_y_x,
    log_base: float = 2,
    thresh: float = 1e-12,
    max_iter: int = 1e3,
    lambda_2: float = 0.1,
) -> tuple:
    """
    Maximize the capacity between I(X;Y)
    p_y_x: each row represnets probability assinmnet
    log_base: the base of the log when calaculating the capacity
    thresh: the threshold of the update, finish the calculation when gettting to it.
    max_iter: the maximum iterations of the calculation
    """

    # Input test
    if torch.abs(torch.mean(torch.sum(p_y_x, axis=1)) - 1) > 1e-6:
        breakpoint()
    assert p_y_x.shape[0] > 1

    # The number of inputs: size of |X|
    m = p_y_x.shape[0]

    # The number of outputs: size of |Y|
    n = p_y_x.shape[1]

    # Initialize the prior uniformly
    r = torch.ones((1, m)) / m

    # Compute the r(x) that maximizes the capacity
    for iteration in range(int(max_iter)):

        q = r.reshape(-1, 1) * p_y_x
        q = q / torch.sum(q, axis=0)

        r1 = torch.prod(torch.pow(q, p_y_x), axis=1)
        r1 = r1 / torch.sum(r1)

        tolerance = torch.linalg.norm(r1 - r)
        r = r1
        if tolerance < thresh:
            break

    return r


def return_pdf_y_x_for_ba(config, regime_class):
    if regime_class.config["regime"] == 1:
        pdf_y_given_x = regime_class.pdf_y_given_v
    elif regime_class.config["regime"] == 2:
        pdf_y_given_x = regime_class.pdf_u_given_x
    elif regime_class.config["regime"] == 3:
        pdf_x = torch.ones(len(regime_class.alphabet_x)) / len(regime_class.alphabet_x)
        pdf_y_given_x = regime_class.calculate_pdf_y_given_x(pdf_x)

    return torch.transpose(pdf_y_given_x, 0, 1)  # TODO: Check this transpose


def apply_blahut_arimoto(regime_class, config):
    # Only Peak Power Constraint works
    print("...Blahut-Arimoto Capacity...")
    pdf_y_given_x = return_pdf_y_x_for_ba(config, regime_class)
    pdf_y_given_x = pdf_y_given_x.numpy()

    capacity, input_dist = blahut_arimoto(
        pdf_y_given_x,
        regime_class.alphabet_x.numpy(),
        log_base=np.e,
        average_power=False,
    )

    print("Blahut Arimoto Capacity: ", capacity)
    return capacity, input_dist


def main_ba():

    # Average Power Constraint - Sweep over lambda_2
    lambda_2 = np.linspace(0.000001, 0.15, 0)
    config = read_config()
    power = config["max_power_cons"]

    # give the biggest alphabet for maximum snr
    alphabet_x, alphabet_y, _, _ = get_alphabet_x_y(config, power)
    regime_class = get_regime_class(config, alphabet_x, alphabet_y, power, config["tanh_factor"])
    pdf_y_given_x = return_pdf_y_x_for_ba(config, regime_class)

    snr_change, noise_power = regime_dependent_snr(config)

    power = []
    capacity = []
    log_snr = []
    snr_range = []
    map_snr_pdf = {}
    for l in lambda_2:
        print("-----------------Lambda_2: ", l, "--------------------")
        cap, input_dist = blahut_arimoto(
            pdf_y_given_x.numpy(), alphabet_x.numpy(), log_base=np.e, lambda_2=l
        )

        p = np.sum(input_dist * alphabet_x.numpy() ** 2)
        log_snr.append(np.log(1 + p / (noise_power)) / 2)
        snr = 10 * np.log10(p / (noise_power))
        snr_range.append(snr)

        map_snr_pdf[str(snr)] = [input_dist, alphabet_x.numpy()]

        print("SNR: ", snr)
        print("Capacity: ", cap)
        print("Log SNR: ", log_snr[-1])
        power.append(p)
        capacity.append(cap)

    file_name = (
        "blahut_arimoto/"
        + "power="
        + str(config["max_power_cons"])
        + "_"
        + "regime="
        + str(config["regime"])
        + "_"
        + "nonlinearity="
        + str(config["nonlinearity"])
    )
    try:
        os.mkdir("blahut_arimoto")
    except FileExistsError:
        pass

    try:
        os.mkdir(file_name)
    except FileExistsError:
        pass

    # No TDM here so I will forcefully mention if arguments file is wrong
    config["time_division_active"] = False

    res = {"BA Capacity": capacity, "Log SNR": log_snr}
    plot_pdf_vs_change(map_snr_pdf, snr_range, config, save_location=file_name)

    plot_vs_change(snr_range, res, config, save_location=file_name)

    res["snr_range"] = snr_range
    io.savemat(file_name + "/results.mat", res)
    io.savemat(file_name + "/pdf_snr.mat", map_snr_pdf)


def main_interference():
    config = read_config(args_name="arguments-interference.yml")
    power = config["max_power_cons"]
    power2 = config["power_2"]
    # Check if the power constraint is average
    if config["cons_type"] == 1:
        average_power = True
    else:
        average_power = False

    alphabet_x_RX1, alphabet_y_RX1, alphabet_x_RX2, alphabet_y_RX2 = (
        get_interference_alphabet_x_y(config, power, power2)
    )
    file_name = (
        "blahut_arimoto-interference/"
        + "power="
        + str(config["max_power_cons"])
        + "_"
        + "regime="
        + str(config["regime"])
        + "_"
        + "nonlinearity="
        + str(config["nonlinearity"])
        + "_cons_type="
        + str(config["cons_type"])
    )
    os.makedirs(file_name, exist_ok=True)
    lambda_2 = np.linspace(0.000001, 0.15, 10)
    lambda_imp = np.linspace(0.1, 0.9, 10)

    power_1 = []
    power_2 = []
    cap_RX1 = []
    cap_RX2 = []
    print("-----------BA for Interference Channel---------")
    print("Average Power: ", average_power)

    # Lambda_2 is only necessary for average power constraint
    if not average_power:
        lambda_2 = [0]
    if average_power:
        lambda_imp = [0]

    for lamb_imp in lambda_imp:
        if not average_power:
            print("-----------------Lambda_Imp: ", lamb_imp, "--------------------")
        for l in lambda_2:
            if average_power:
                print("-----------------Lambda_2: ", l, "--------------------")
            cap1, pdf_x1, cap2, pdf_x2 = blahut_arimoto_interference(
                alphabet_x_RX1,
                alphabet_y_RX1,
                alphabet_x_RX2,
                alphabet_y_RX2,
                config,
                lambda_2=l,
                average_power=average_power,
                lambda_importance=lamb_imp,
            )
            cap_RX1.append(cap1)
            cap_RX2.append(cap2)

            # Lets look at the power only - SNR definition is confusing here
            power_1.append(torch.sum(pdf_x1 * alphabet_x_RX1**2))
            power_2.append(torch.sum(pdf_x2 * alphabet_x_RX2**2))

            print("Power 1: ", power_1)
            print("Power 2: ", power_2)

    # Plot RX1
    # breakpoint()
    if average_power:
        plt.plot(power_1, cap_RX1, label="RX1")
        plt.plot(power_2, cap_RX2, label="RX2")
        plt.xlabel("Power")
        plt.ylabel("Capacity")
        plt.legend()
        plt.savefig(file_name + "/interference_capacity.png")
        # plt.show()

    res = {
        "cap_RX1": cap_RX1,
        "cap_RX2": cap_RX2,
        "power_1": power_1,
        "power_2": power_2,
    }
    io.savemat(file_name + "/results.mat", res)

    if not average_power:
        res_plot = {"R1": {}, "R2": {}}
        res_plot["R1"]["BA"] = cap_RX1
        res_plot["R2"]["BA"] = cap_RX2
        plot_R1_R2_curve(res_plot, power, power2, file_name)
    else:
        print("HOW DO WE PLOT R1 AND R2 FOR AVERAGE POWER CONSTRAINT?")
    print("Results saved at: ", file_name)


def blahut_arimoto_interference(
    alphabet_x_RX1,
    alphabet_y_RX1,
    alphabet_x_RX2,
    alphabet_y_RX2,
    config,
    average_power=True,
    log_base: float = np.e,
    thresh: float = 1e-12,
    max_iter: int = 1e3,
    lambda_2: float = 0.1,
    lambda_importance: float = 0.5,
) -> tuple:

    if config["regime"] != 1:
        raise ValueError("Only first regime is implemented")

    # First channel's regime class defined
    config["sigma_2"] = config["sigma_12"]
    regime_RX1 = First_Regime(alphabet_x_RX1, alphabet_y_RX1, config)
    config["sigma_2"] = config["sigma_22"]
    regime_RX2 = First_Regime(alphabet_x_RX2, alphabet_y_RX2, config)
    pdf_y_given_x_RX2 = torch.transpose(
        regime_RX2.pdf_y_given_v, 0, 1
    )  # point to point
    # RX1 pdf_y_given_x will change depending on the RX2 input since there is interference

    # Input test
    check_input(pdf_y_given_x_RX2)
    # The number of inputs: size of |X|
    m_RX1 = alphabet_x_RX1.shape[0]
    m_RX2 = alphabet_x_RX2.shape[0]

    # The number of outputs: size of |Y|
    n_RX1 = alphabet_y_RX1.shape[0]
    n_RX2 = alphabet_y_RX2.shape[0]

    # Initialize the prior uniformly
    pdf_x1 = torch.ones((1, m_RX1)) / m_RX1
    pdf_x2 = torch.ones((1, m_RX2)) / m_RX2

    # Compute the r(x) that maximizes the capacity
    for iteration in range(int(max_iter)):

        # First optimize for RX2 since that one does not have interference
        q_RX2 = pdf_x2.reshape(-1, 1) * pdf_y_given_x_RX2
        q_RX2 = q_RX2 / torch.sum(q_RX2, axis=0)

        if average_power:
            pdf_2 = torch.prod(torch.pow(q_RX2, pdf_y_given_x_RX2), axis=1) * torch.exp(
                -lambda_2 * (alphabet_x_RX2**2)
            )
        else:  # Peak Power
            pdf_2 = torch.prod(torch.pow(q_RX2, pdf_y_given_x_RX2), axis=1)
        pdf_2 = pdf_2 / torch.sum(pdf_2).to(torch.float32)
        tolerance_2 = torch.linalg.norm(pdf_2 - pdf_x2)
        pdf_x2 = pdf_2

        regime_RX1.set_interference_active(alphabet_x_RX2, pdf_x2)
        regime_RX1.set_pdf_u(pdf_x1.flatten())
        pdf_y_given_x_RX1 = torch.transpose(
            regime_RX1.calculate_pdf_y_given_x_interference(), 0, 1
        )
        check_input(pdf_y_given_x_RX1)

        q_RX1 = pdf_x1.reshape(-1, 1) * pdf_y_given_x_RX1
        q_RX1 = q_RX1 / torch.sum(q_RX1, axis=0)

        if average_power:
            pdf_1 = torch.prod(torch.pow(q_RX1, pdf_y_given_x_RX1), axis=1) * torch.exp(
                -lambda_2 * (alphabet_x_RX1**2)
            )
        else:  # Peak Power
            pdf_1 = torch.prod(torch.pow(q_RX1, pdf_y_given_x_RX1), axis=1)
        pdf_1 = pdf_1 / torch.sum(pdf_1).to(torch.float32)

        tolerance_1 = torch.linalg.norm(pdf_1 - pdf_x1)
        pdf_x1 = pdf_1

        # If both is not changing much, break
        # if tolerance_2 < thresh and tolerance_1 < thresh:
        #     break

        if (
            lambda_importance * tolerance_1 + (1 - lambda_importance) * tolerance_2
            < thresh
        ):
            break

        if iteration % 100 == 0:
            # Calculate the capacity
            cap1 = regime_RX1.capacity(pdf_x1.flatten())
            cap2 = regime_RX2.capacity(pdf_x2.flatten())
            print("Iteration: ", iteration, " Cap1: ", cap1, " Cap2: ", cap2)

    # Calculate the capacity
    cap1 = regime_RX1.capacity(pdf_x1.flatten())
    cap2 = regime_RX2.capacity(pdf_x2.flatten())

    return cap1, pdf_x1, cap2, pdf_x2


def check_input(pdf_y_given_x):
    assert np.abs(pdf_y_given_x.sum(axis=1).mean() - 1) < 1e-6
    assert pdf_y_given_x.shape[0] > 1


if __name__ == "__main__":
    # main_ba()
    main_interference()
