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
    generate_alphabet_x_y,
    return_regime_class,
    regime_dependent_snr,
    plot_pdf_snr,
    plot_snr,
)
import torch
import matplotlib.pyplot as plt
import os
from scipy import io


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


def return_pdf_y_x_for_ba(config, regime_class):

    if regime_class.config["regime"] == 1:
        pdf_y_given_x = regime_class.calculate_pdf_y_given_v()
    elif regime_class.config["regime"] == 2:
        pdf_y_given_x = regime_class.calculate_pdf_u_given_x()
    elif regime_class.config["regime"] == 3:
        pdf_x = torch.ones(len(regime_class.alphabet_x)) / len(regime_class.alphabet_x)
        pdf_y_given_x = regime_class.calculate_pdf_y_given_x(pdf_x)

    pdf_y_given_x = np.transpose(pdf_y_given_x.numpy())
    return pdf_y_given_x


def apply_blahut_arimoto(regime_class, config):
    # Only Peak Power Constraint works
    print("---------Blahut-Arimoto Capacity---------")
    pdf_y_given_x = return_pdf_y_x_for_ba(config, regime_class)

    capacity, input_dist = blahut_arimoto(
        pdf_y_given_x,
        regime_class.alphabet_x.numpy(),
        log_base=np.e,
        average_power=False,
    )

    print("Blahut Arimoto Capacity: ", capacity)
    return capacity, input_dist


def main():

    # Average Power Constraint - Sweep over lambda_2
    lambda_2 = np.linspace(0.005, 0.15, 30)
    config = read_config()
    power = config["max_power_cons"]

    # give the biggest alphabet for maximum snr
    alphabet_x, alphabet_y, _, _ = generate_alphabet_x_y(config, power)
    regime_class = return_regime_class(config, alphabet_x, alphabet_y, power)
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
            pdf_y_given_x, alphabet_x.numpy(), log_base=np.e, lambda_2=l
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

    res = {"BA Capacity": capacity, "Log SNR": log_snr}
    plot_pdf_snr(map_snr_pdf, snr_range, config, save_location=file_name)

    plot_snr(snr_range, res, config, save_location=file_name)

    res["snr_range"] = snr_range
    io.savemat(file_name + "/results.mat", res)
    io.savemat(file_name + "/pdf_snr.mat", map_snr_pdf)


if __name__ == "__main__":
    main()
