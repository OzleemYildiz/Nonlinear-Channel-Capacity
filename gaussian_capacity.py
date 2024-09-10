from utils import loss
import torch


def gaussian_capacity(regime_class):
    # print("Gaussian Capacity Calculation")
    pdf_x = (
        1
        / (torch.sqrt(torch.tensor([2 * torch.pi * regime_class.power])))
        * torch.exp(
            -0.5 * ((regime_class.alphabet_x) ** 2) / regime_class.power
        ).float()
    )
    pdf_x = (pdf_x / torch.sum(pdf_x)).to(torch.float32)

    loss_g = loss(
        pdf_x,
        regime_class,
    )

    cap_g = -loss_g

    print("Gaussian Capacity: ", cap_g)
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
