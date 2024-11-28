import numpy as np
import torch
from nonlinearity_utils import return_nonlinear_fn


def get_capacity_PP(config, pdf_x, alphabet_x, alphabet_y):

    if config["regime"] == 1:
        cap = get_first_regime_capacity(config, pdf_x, alphabet_x, alphabet_y)
    return cap


def get_capacity_interference():
    pass


def get_first_regime_capacity(config, pdf_x, alphabet_x, alphabet_y):
    func = return_nonlinear_fn(config)
    alphabet_v = func(alphabet_x)

    pdf_y_given_x = (
        1
        / (torch.sqrt(torch.tensor([2 * torch.pi])) * config["sigma_2"])
        * torch.exp(
            -0.5
            * ((alphabet_y.reshape(-1, 1) - alphabet_v.reshape(1, -1)) ** 2)
            / config["sigma_2"] ** 2
        )
    )

    pdf_y_given_x = pdf_y_given_x / (torch.sum(pdf_y_given_x, axis=0) + 1e-30)
    py_x_logpy_x = pdf_y_given_x * torch.log(pdf_y_given_x + 1e-20)
    px_py_x_logpy_x = py_x_logpy_x @ pdf_x
    f_term = torch.sum(px_py_x_logpy_x)
    py = pdf_y_given_x @ pdf_x
    s_term = torch.sum(py * torch.log(py + 1e-20))
    return f_term - s_term


def get_third_regime_capacity(config, pdf_x, alphabet_x, alphabet_y):
    pdf_y = torch.zeros_like(pdf_x)
