from utils import loss
import torch
from First_Regime import First_Regime
import math


def gaussian_capacity(regime_class, power):
    # print("Gaussian Capacity Calculation")
    pdf_x = (
        1
        / (torch.sqrt(torch.tensor([2 * torch.pi * power])))
        * torch.exp(-0.5 * ((regime_class.alphabet_x) ** 2) / power).float()
    )
    pdf_x = (pdf_x / torch.sum(pdf_x)).to(torch.float32)

    loss_g = loss(
        pdf_x,
        regime_class,
        # project_active=False,
    )

    cap_g = -loss_g

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
    power1,
    power2,
    config,
    alphabet_x_RX1,
    alphabet_y_RX1,
    alphabet_x_RX2,
    alphabet_y_RX2,
):
    # Second User does not have interference
    pdf_x_2 = (
        1
        / (torch.sqrt(torch.tensor([2 * torch.pi * power2])))
        * torch.exp(-0.5 * ((alphabet_x_RX2) ** 2) / power2).float()
    )
    pdf_x_2 = (pdf_x_2 / torch.sum(pdf_x_2)).to(torch.float32)
    config["sigma_2"] = config["sigma_22"]
    f_reg_RX2 = First_Regime(alphabet_x_RX2, alphabet_y_RX2, config, power2)  # 2nd user
    cap_RX2 = f_reg_RX2.new_capacity(pdf_x_2)

    # Z = X1 +aX2 distribution has variance power + a^2 power
    # TODO: This is bad way to solve it. Need to find a better way

    pdf_x_1 = (
        1
        / (torch.sqrt(torch.tensor([2 * torch.pi * power1])))
        * torch.exp(-0.5 * ((alphabet_x_RX1) ** 2) / power1).float()
    )
    # breakpoint()
    pdf_x_1 = (pdf_x_1 / torch.sum(pdf_x_1)).to(torch.float32)
    config["sigma_2"] = config["sigma_12"]
    f_reg_RX1 = First_Regime(alphabet_x_RX1, alphabet_y_RX1, config, power1)  # 1st user
    cap_RX1 = f_reg_RX1.capacity_of_interference(
        pdf_x_1,
        pdf_x_2,
        f_reg_RX2.alphabet_x,
    )

    return cap_RX1, cap_RX2


def find_best_gaussian(regime_class):
    max_power = regime_class.power

    power_range = torch.arange(max_power, 0, -1)
    best_cap = 0
    best_power = 0
    for power in power_range:
        cap_g = gaussian_capacity(regime_class, power)
        if cap_g - best_cap > 1e-5:
            best_cap = cap_g
            best_power = power
        else:
            # lowering the power at every step but it does not increase the capacity
            # then higher the power, the better
            break
    print("Gaussian Capacity: ", best_cap)
    return best_power, best_cap
