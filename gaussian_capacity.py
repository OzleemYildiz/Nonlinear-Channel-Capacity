from utils import (
    loss,
    get_interference_alphabet_x_y,
    get_regime_class_interference,
    loss_interference,
)
import torch
from First_Regime import First_Regime
import math
import numpy as np
from nonlinearity_utils import return_nonlinear_fn


def gaussian_capacity(regime_class, power, complex_alphabet=False):
    # print("Gaussian Capacity Calculation")
    pdf_x = get_gaussian_distribution(power, regime_class, complex_alphabet)

    loss_g = loss(
        pdf_x,
        regime_class,
        # project_active=False,
    )

    cap_g = -loss_g
    # print("~~~~~Gaussian Capacity:", cap_g, "~~~~~")

    return cap_g


def gaussian_with_l1_norm(alphabet_x, alphabet_y, power, config):
    print("Gaussian Capacity Calculation with L1 Norm Constraint")
    gaus_power = (torch.pi / 2) * power**2
    pdf_x = (
        1
        / (torch.sqrt(torch.tensor([2 * torch.pi * gaus_power])))
        * torch.exp(-0.5 * ((alphabet_x) ** 2) / gaus_power)
    )
    pdf_x = pdf_x / torch.sum(pdf_x).to(torch.float32)
    loss_r = loss(
        pdf_x,
        alphabet_y,
        alphabet_x,
        power,
        config,
    )
    cap_r = capacity(loss_r, config, alphabet_x, pdf_x, power)
    return cap_r


def gaussian_interference_capacity(
    reg_RX1, reg_RX2, int_ratio, tin_active, pdf_x_RX2=None
):
    # Second User does not have interference

    pdf_x_1 = get_gaussian_distribution(
        reg_RX1.power, reg_RX1, complex_alphabet=reg_RX1.config["complex"]
    )

    if pdf_x_RX2 is None:
        pdf_x_RX2 = get_gaussian_distribution(
            reg_RX2.power, reg_RX2, complex_alphabet=reg_RX2.config["complex"]
        )
    pdf_x_RX2 = pdf_x_RX2.detach()

    loss, cap_RX1, cap_RX2 = loss_interference(
        pdf_x_1, pdf_x_RX2, reg_RX1, reg_RX2, int_ratio, tin_active, lmbd=0.5
    )

    return cap_RX1, cap_RX2


def find_best_gaussian(regime_class, complex_alphabet=False):
    max_power = regime_class.power
    try:
        power_range = torch.arange(max_power, 1, -1)
    except:
        return max_power, gaussian_capacity(regime_class, max_power, complex_alphabet)
    best_cap = 0
    best_power = max_power
    ind = 0
    for power in power_range:
        cap_g = gaussian_capacity(regime_class, power, complex_alphabet)
        if cap_g > best_cap:
            best_cap = cap_g
            best_power = power
            ind = 0
        else:
            if ind > 100:
                # lowering the power at every step but it does not increase the capacity
                # then higher the power, the better
                break
        ind += 1

    print("Gaussian Capacity: ", best_cap)
    return best_power, best_cap


# TODO: Not done
def agc_gaussian_capacity_interference(config, power):
    power1 = power
    power2 = config["power_2"]
    power_max = config["clipping_limit_x"]
    noise_power = config["sigma_12"] ** 2  # Noise for the first receiver

    alpha = np.sqrt(min(1, (power_max - noise_power) / (power1 + power2)))

    # So the system model was Y = phi(X_1+X_2)+N_2
    # Now we change this to Y = phi(alpha*(X_1+X_2))+N_2

    alphabet_x_RX1, alphabet_y_RX1, alphabet_x_RX2, alphabet_y_RX2 = (
        get_interference_alphabet_x_y(config, power, power2)
    )
    alphabet_x_RX1 = alpha * alphabet_x_RX1
    alphabet_x_RX2 = alpha * alphabet_x_RX2
    # breakpoint()
    cap1, cap2 = gaussian_interference_capacity(
        power1,
        power2,
        config,
        alphabet_x_RX1,
        alphabet_y_RX1,
        alphabet_x_RX2,
        alphabet_y_RX2,
    )

    return cap1, cap2


def update_nonlinear_function(config, alpha):
    pass


def gaus_interference_R1_R2_curve(config, power, power2):
    cap_gaus_RX1 = []
    cap_gaus_RX2 = []
    ratio_pw = np.linspace(0.01, 0.99, 20)
    for ratio in ratio_pw:
        power1 = power
        power2_upd = ratio * power2

        # print("-----------Power: ", power, "when lambda=", lmbd, "-----------")
        alphabet_x_RX1, alphabet_y_RX1, alphabet_x_RX2, alphabet_y_RX2 = (
            get_interference_alphabet_x_y(config, power, power2_upd)
        )
        cap1, cap2 = gaussian_interference_capacity(
            power1,
            power2_upd,
            config,
            alphabet_x_RX1,
            alphabet_y_RX1,
            alphabet_x_RX2,
            alphabet_y_RX2,
        )
        cap_gaus_RX1.append(cap1.numpy().item())
        cap_gaus_RX2.append(cap2.numpy().item())
    return cap_gaus_RX1, cap_gaus_RX2


def get_gaussian_distribution(
    power, regime_class, complex_alphabet=False, optimum_power=False
):
    if optimum_power:
        (power, _) = find_best_gaussian(regime_class, complex_alphabet)
        print("Optimum Power: ", power)
        # power = 75
    if complex_alphabet:
        pdf_x_re = (
            1
            / (torch.sqrt(torch.tensor([2 * torch.pi * power / 2])))
            * torch.exp(
                -0.5 * ((regime_class.alphabet_x_re) ** 2) / (power / 2)
            ).float()
        )
        pdf_x_re = (pdf_x_re / torch.sum(pdf_x_re)).to(torch.float32)

        pdf_x_imag = (
            1
            / (torch.sqrt(torch.tensor([2 * torch.pi * power / 2])))
            * torch.exp(
                -0.5 * ((regime_class.alphabet_x_im) ** 2) / (power / 2)
            ).float()
        )
        pdf_x_imag = (pdf_x_imag / torch.sum(pdf_x_imag)).to(torch.float32)

        pdf_x = pdf_x_re * pdf_x_imag
        pdf_x = (pdf_x / torch.sum(pdf_x)).to(torch.float32)
        pdf_x = pdf_x.reshape(-1)

    else:
        pdf_x = (
            1
            / (torch.sqrt(torch.tensor([2 * torch.pi * power])))
            * torch.exp(-0.5 * ((regime_class.alphabet_x_re) ** 2) / power).float()
        )
        pdf_x = (pdf_x / torch.sum(pdf_x)).to(torch.float32)

    return pdf_x
