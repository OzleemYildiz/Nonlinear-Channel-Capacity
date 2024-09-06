import argparse
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp
import os
from scipy.integrate import quad
from scipy.stats import differential_entropy, entropy
import random


def sample_x_from_dist(alphabet_x, pdf_x, power):
    x_samples = torch.tensor(
        [random.gauss(0, np.sqrt(power)) for _ in range(len(alphabet_x))]
    )
    return x_samples


def joint_entropies(data, nbins=None):
    n_variables = data.shape[-1]
    n_samples = data.shape[0]
    if nbins == None:
        nbins = int((n_samples / 5) ** 0.5)
    histograms2d = np.zeros((n_variables, n_variables, nbins, nbins))
    for i in range(n_variables):
        for j in range(n_variables):
            histograms2d[i, j] = np.histogram2d(data[:, i], data[:, j], bins=nbins)[0]
    probs = histograms2d / len(data) + 1e-100
    joint_entropies = -(probs * np.log2(probs)).sum((2, 3))
    return joint_entropies


def mutual_info_matrix(data, nbins=None, normalized=True):
    n_variables = data.shape[-1]
    j_entropies = joint_entropies(data, nbins)
    entropies = j_entropies.diagonal()
    entropies_tile = np.tile(entropies, (n_variables, 1))
    sum_entropies = entropies_tile + entropies_tile.T
    mi_matrix = sum_entropies - j_entropies
    if normalized:
        mi_matrix = mi_matrix * 2 / sum_entropies
    return mi_matrix


def capacity_try(alphabet_x, pdf_x, config, power):
    nonlinear_func = return_nonlinear_fn(config)
    x_samples = sample_x_from_dist(alphabet_x, pdf_x, power)

    if config["regime"] == 1:
        y_after_nonlinear = nonlinear_func(x_samples) + torch.tensor(
            [random.gauss(0, config["sigma"]) for _ in range(len(x_samples))]
        )
    elif config["regime"] == 2:
        y_after_nonlinear = nonlinear_func(
            x_samples
            + torch.tensor(
                [random.gauss(0, config["sigma"]) for _ in range(len(x_samples))]
            )
        )
    return mutual_info_matrix(
        torch.cat((x_samples.unsqueeze(1), y_after_nonlinear.unsqueeze(1)), 1)
        .detach()
        .numpy(),
        nbins=100,
    )[0, 1]


def h_y_from_samples(alphabet_x, pdf_x, config, power):
    nonlinear_func = return_nonlinear_fn(config)
    x_samples = sample_x_from_dist(alphabet_x, pdf_x, power)

    if config["regime"] == 1:
        y_after_nonlinear = nonlinear_func(x_samples) + torch.tensor(
            [random.gauss(0, config["sigma"]) for _ in range(len(x_samples))]
        )
    elif config["regime"] == 2:
        y_after_nonlinear = nonlinear_func(
            x_samples
            + torch.tensor(
                [random.gauss(0, config["sigma"]) for _ in range(len(x_samples))]
            )
        )

    h_y = differential_entropy(y_after_nonlinear)
    print("h_y:", h_y)
    return h_y


def h_y_given_x_from_samples(alphabet_x, pdf_x, config):
    nonlinear_func = return_nonlinear_fn(config)
    h_y_given_x = 0
    for ind, x in enumerate(alphabet_x):
        if config["regime"] == 1:
            y = nonlinear_func(x) + torch.tensor(
                [random.gauss(0, config["sigma"]) for _ in range(config["n_samples"])]
            )
            h_y_given_x += differential_entropy(y) * pdf_x[alphabet_x.index(x)]
        elif config["regime"] == 2:
            y = nonlinear_func(
                x
                + torch.tensor(
                    [
                        random.gauss(0, config["sigma"])
                        for _ in range(config["n_samples"])
                    ]
                )
            )
            h_y_given_x += differential_entropy(y) * pdf_x[ind]
    print("h_y_given_x:", h_y_given_x)
    return h_y_given_x


def capacity_from_samples(alphabet_x, pdf_x, config, power):
    h_y = h_y_from_samples(alphabet_x, pdf_x, config, power)
    h_y_given_x = h_y_given_x_from_samples(alphabet_x, pdf_x, config)
    cap = h_y - h_y_given_x
    return cap


# calculate the conditional pdf of X given std, stop_std and peak_power
def cond_yx(alphabet_x, y, config):

    cond_yx = (
        1
        / (torch.sqrt(torch.tensor([2 * torch.pi])) * config["sigma"])
        * torch.exp(-0.5 * ((y - alphabet_x) ** 2) / config["sigma"] ** 2)
    )

    # nonlinearity is after channel output, e.g., y = phi(x+N)
    # This won't work if it's not one-to-one function
    if config["regime"] == 2:
        d_nonlinear = return_derivative_of_nonlinear_fn(config)
        denom = torch.abs(d_nonlinear(alphabet_x))
        cond_yx = cond_yx / denom

    return cond_yx


def new_hy_x(cond_yx, alphabet_y, alphabet_x, config, pdf_x):
    h_y_x = 0
    for y in alphabet_y:
        cond_yx_pdf = cond_yx(alphabet_x, y, config)
        h_y_x += pdf_x @ (
            cond_yx_pdf * config["delta_y"] * torch.log(cond_yx_pdf + 1e-30)
        )
    return h_y_x


def project_pdf(pdf_x, cons_type, alphabet_x, power):
    # breakpoint()
    # pdf cannot be negative
    # pdf_x = torch.relu(pdf_x)
    # sum of pdf is 1
    # pdf_x = pdf_x/torch.sum(pdf_x)
    # average power constraint
    n, m = len(alphabet_x), 1
    if n == 0:
        breakpoint()
    p_hat = cp.Variable(n)
    p = cp.Parameter(n)
    A = cp.Parameter((m, n))
    mu = cp.Parameter(m)
    if cons_type == 1:
        # average power is in theconstraint
        constraints = [p_hat >= 0, cp.sum(p_hat) == 1, A @ p_hat <= mu]
    elif cons_type == 0:  # peak power
        constraints = [p_hat >= 0, cp.sum(p_hat) == 1]
    else:
        constraints = [p_hat >= 0, cp.sum(p_hat) == 1, A @ p_hat <= mu]

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
    alphabet_y,
    alphabet_x,
    power,
    config,
):
    # Projection is done with the main alphabet
    if len(alphabet_x) == 0:
        breakpoint()
        aaa = 1

    pdf_x = project_pdf(pdf_x, config["cons_type"], alphabet_x, power)
    h_y = 0
    change_alphabet_x = alphabet_fix_nonlinear(alphabet_x, config)

    p_y = []
    for y in alphabet_y:
        cond_yx_pdf = cond_yx(change_alphabet_x, y, config)
        # breakpoint()
        if pdf_x @ cond_yx_pdf > 0:
            h_y += (
                pdf_x @ cond_yx_pdf * config["delta_y"] * torch.log(pdf_x @ cond_yx_pdf)
            )
        # p_y = torch.cat((p_y, torch.tensor([pdf_x@cond_yx_pdf])))
        # p_y.append(pdf_x@cond_yx_pdf)
    # p_y = torch.tensor(p_y)s
    # breakpoint()
    # py_norm = p_y/torch.sum(p_y)
    # breakpoint()
    # h_y = torch.sum(torch.log(p_y) * py_norm)
    # h_y = torch.tensor(p_y)@torch.log(torch.tensor(p_y))
    if torch.isnan(h_y):
        breakpoint()

    return h_y  # + torch.log(torch.tensor([delta_y]))


def capacity(loss, config, alphabet_x, pdf_x, power, alphabet_y):
    pdf_x = project_pdf(pdf_x, config["cons_type"], alphabet_x, power)
    alphabet_x = alphabet_fix_nonlinear(alphabet_x, config)

    # TODO: Fix this
    if config["regime"] == 1:  # or config["nonlinearity"] == 0:

        cap = -loss - 0.5 * torch.log(
            torch.tensor([2 * np.pi * np.e * config["sigma"] ** 2])
        )
    elif config["regime"] == 2:

        # hy_x = entropy_y_given_x(alphabet_x, config, pdf_x)
        n_hy_x = new_hy_x(cond_yx, alphabet_y, alphabet_x, config, pdf_x)
        print("hy_x:", n_hy_x)
        print("loss:", loss)
        cap = -loss + n_hy_x
        print("Capacity:", cap)
    return cap


def entropy_y_given_x(alphabet_x, config, pdf_x):
    # TODO: Can I just calculate the integral -- Could be slow but better than nothing

    d_nonlinear = return_derivative_of_nonlinear_fn(config)
    # H(Y|X) = 1/(sigma*sqrt(2*pi)) * (A+B+C)

    A = (
        torch.sqrt(torch.tensor(2 * torch.pi))
        * config["sigma"]
        / 2
        * torch.log(torch.tensor(2 * torch.pi * config["sigma"] ** 2))
    )
    B = torch.sqrt(torch.tensor(2 * torch.pi)) * config["sigma"] / 2
    C = 0
    h_y_given_x = 0
    eps = 1e-30
    print("A:", A)
    print("B:", B)

    multiplier = 1 / (config["sigma"] * torch.sqrt(torch.tensor(2 * torch.pi)))

    for ind, x in enumerate(alphabet_x):
        integrand_c = lambda u: torch.exp(
            torch.tensor(-0.5 * (u - x) ** 2 / config["sigma"] ** 2)
        ) * torch.log(torch.abs(d_nonlinear(u)) + eps)
        C_integrand = quad(integrand_c, -torch.inf, torch.inf)
        if C_integrand[1] > 1e-3:
            print("Error BIG:", C_integrand[1])

        C = C_integrand[0]  # * torch.log(
        #     torch.tensor(2 * torch.pi * config["sigma"] ** 2) * 0.5)

        h_y_given_x += (A + B + C) * pdf_x[ind] * multiplier

    print("Error:", C_integrand[1])
    print("C:", C)
    return h_y_given_x


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
):

    plt.figure(figsize=(5, 4))
    leg_str = []
    for keys in res.keys():
        plt.plot(snr_range, res[keys])
        leg_str.append(keys)
    plt.legend(leg_str)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Capacity")
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
    plt.close()


def plot_pdf_snr(map_snr_pdf, snr_range, config):
    breakpoint()
    plt.figure(figsize=(5, 4))
    for snr in snr_range:
        pdf_x, alphabet_x = map_snr_pdf[snr]
        act_pdf_x = pdf_x.detach().numpy() > 0
        act_alp = alphabet_x.detach().numpy()[act_pdf_x]
        plt.scatter(
            snr * np.ones_like(act_alp),
            act_alp,
            s=pdf_x.detach().numpy()[act_pdf_x] * 100,
        )
    plt.xlabel("SNR (dB)")
    plt.ylabel("X")
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
    plt.close()


# It is one-one function so pdf stays the alphabet changes
def alphabet_fix_nonlinear(alphabet_x, config):
    if config["regime"] == 1:
        nonlinear_func = return_nonlinear_fn(config)
        x_change = nonlinear_func(alphabet_x)
    elif config["regime"] == 2:
        x_change = alphabet_x  # since nonlinearity is applied to the channel output
    return x_change


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

    # phi(X)+ N
    if config["regime"] == 1:
        # 0:linear
        max_y = nonlinear_func(max_x) + config["sigma"] ** 2

    elif config["regime"] == 2:  # phi(X+N)
        # NOTE!!: We will only use 0,3 and 4 nonlinearity for now -- 1 and 2 are not supported
        if config["nonlinearity"] == 0:  # linear
            max_y = max_x + config["sigma"] ** 2
        elif config["nonlinearity"] == 3:
            max_y = 1  # since tanh is defined between -1 and 1
        elif config["nonlinearity"] == 4:
            max_y = 1  # since x/(1+x^4)^1/4 is defined between -1 and 1

    # alphabet_y

    # sample_num = math.ceil( 2*stop_s*(np.sqrt(max_y)+ sigma) /delta_y)
    # alphabet_y = torch.linspace(-stop_s*(np.sqrt(max_y)+sigma), stop_s*(np.sqrt(max_y)+sigma) ,sample_num )

    ## DOUBLE CHECK???
    if config["regime"] == 1 or config["nonlinearity"] == 0:
        sample_num = math.ceil(2 * config["stop_sd"] * (max_y) / config["delta_y"])
        alphabet_y = torch.linspace(
            -config["stop_sd"] * (max_y), config["stop_sd"] * (max_y), sample_num
        )
    elif config["regime"] == 2:
        if config["nonlinearity"] == 3:
            sample_num = math.ceil(2 * max_y / config["delta_y"])
            alphabet_y = torch.linspace(-max_y, max_y, sample_num)
        elif config["nonlinearity"] == 4:  # even function
            sample_num = math.ceil(max_y / config["delta_y"])
            alphabet_y = torch.linspace(0, max_y, sample_num)

    # Gaussian Result
    # alphabet_x
    sample_num_g = math.ceil(2 * max_x / config["delta_y"])
    alphabet_x = torch.linspace(-max_x, max_x, sample_num_g)

    return alphabet_x, alphabet_y, max_x, max_y


def return_nonlinear_fn(config):
    # 0:linear
    if config["nonlinearity"] == 0:
        nonlinear_fn = lambda x: x
    #  1:nonlinear C1 (sgn(X)sqrt(abs(X)))
    elif config["nonlinearity"] == 1:
        nonlinear_fn = lambda x: torch.sign(torch.tensor(x)) * torch.sqrt(
            torch.abs(torch.tensor(x))
        )
    # 2: nonlinear C2 (sgn(X)abs(X)^{0.9})
    elif config["nonlinearity"] == 2:
        nonlinear_fn = (
            lambda x: torch.sign(torch.tensor(x)) * torch.abs(torch.tensor(x)) ** 0.9
        )
    # 3:nonlinear tanh(X)
    elif config["nonlinearity"] == 3:
        nonlinear_fn = lambda x: torch.tanh(torch.tensor(x / config["tanh_factor"]))
    # 4:nonlinear x4: x/(1 + x^4)^1/4
    elif config["nonlinearity"] == 4:
        nonlinear_fn = lambda x: torch.tensor(x / (1 + x**4) ** 0.25)
    else:
        raise ValueError("Nonlinearity not defined")

    return nonlinear_fn


def return_derivative_of_nonlinear_fn(config):
    # 0:linear
    if config["nonlinearity"] == 0:
        nonlinear_fn = lambda x: torch.tensor(1)
    # 3:nonlinear tanh(X)
    elif config["nonlinearity"] == 3:
        nonlinear_fn = (
            lambda x: (1 / torch.cosh(torch.tensor(x / config["tanh_factor"]))) ** 2
            / config["tanh_factor"]
        )
    # 4:nonlinear x4: x/(1 + x^4)^1/4
    elif config["nonlinearity"] == 4:
        nonlinear_fn = lambda x: torch.tensor(1 / (1 + x**4) ** (5 / 4))
    else:
        raise ValueError("Derivative is not supported")

    return nonlinear_fn


# # https://datascience.stackexchange.com/questions/58565/conditional-entropy-calculation-in-python-hyx
# ##Entropy
# def entropy(Y):
#     """
#     Also known as Shanon Entropy
#     Reference: https://en.wikipedia.org/wiki/Entropy_(information_theory)
#     """
#     unique, count = np.unique(Y, return_counts=True, axis=0)
#     prob = count / len(Y)
#     en = np.sum((-1) * prob * np.log2(prob))
#     return en


# # Joint Entropy
# def jEntropy(Y, X):
#     """
#     H(Y;X)
#     Reference: https://en.wikipedia.org/wiki/Joint_entropy
#     """
#     YX = np.c_[Y, X]
#     return entropy(YX)


# # Conditional Entropy
# def cEntropy(Y, X):
#     """
#     conditional entropy = Joint Entropy - Entropy of X
#     H(Y|X) = H(Y;X) - H(X)
#     Reference: https://en.wikipedia.org/wiki/Conditional_entropy
#     """
#     h_xy = jEntropy(Y, X)
#     h_x = entropy(X)
#     print("Joint Entropy:", h_xy)
#     print("Entropy X:", h_x)
#     return h_xy - h_x


# # Information Gain
# def gain(Y, X, config):
#     """
#     Information Gain, I(Y;X) = H(Y) - H(Y|X)
#     Reference: https://en.wikipedia.org/wiki/Information_gain_in_decision_trees#Formal_definition
#     """
#     h_y = entropy(Y)
#     h_y_x = cEntropy(Y, X)
#     print("Entropy Y:", h_y)
#     print("Entropy Y|X:", h_y_x)

#     return h_y - h_y_x
