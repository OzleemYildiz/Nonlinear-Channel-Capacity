from utils import loss, capacity, capacity_from_samples, capacity_try
import torch


def gaussian_capacity(alphabet_x, alphabet_y, power, config):
    # print("Gaussian Capacity Calculation")
    pdf_x = (
        1
        / (torch.sqrt(torch.tensor([2 * torch.pi * power])))
        * torch.exp(-0.5 * ((alphabet_x) ** 2) / power).float()
    )
    pdf_x = (pdf_x / torch.sum(pdf_x)).to(torch.float32)

    # loss_g = loss(
    #     pdf_x,
    #     alphabet_y,
    #     alphabet_x,
    #     power,
    #     config,
    # )
    # # TODO: alphabet_x - make sure
    # cap_g = capacity(loss_g, config, alphabet_x, pdf_x, power, alphabet_y)

    # cap_g = capacity_from_samples(alphabet_x, pdf_x, config, power)

    cap_g = capacity_try(alphabet_x, pdf_x, config, power)

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
