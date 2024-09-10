import torch
from nonlinearity_utils import alphabet_fix_nonlinear
import numpy as np
import math
from nonlinearity_utils import return_nonlinear_fn


class Second_Regime:
    def __init__(self, alphabet_x, config, power):
        self.alphabet_x = alphabet_x
        self.config = config
        self.alphabet_u = self.calculate_u_points()  # U = X+Z_1
        self.pdf_u_given_x = self.calculate_pdf_u_given_x()
        self.power = power
        self.nonlinear_fn = return_nonlinear_fn(self.config)

    def calculate_u_points(self):
        max_u = max(self.alphabet_x) + self.config["sigma_1"] * self.config["stop_sd"]
        sample_num = math.ceil(2 * (max_u) / self.config["delta_y"]) + 1
        alphabet_u = torch.linspace(-(max_u), (max_u), sample_num)
        return alphabet_u

    def calculate_pdf_u_given_x(self):
        pdf_u_given_x = (
            1
            / (torch.sqrt(torch.tensor([2 * torch.pi])) * self.config["sigma_1"])
            * torch.exp(
                -0.5
                * (
                    (self.alphabet_u.reshape(-1, 1) - self.alphabet_x.reshape(1, -1))
                    ** 2
                )
                / self.config["sigma_1"] ** 2
            )
        )
        pdf_u_given_x = pdf_u_given_x / torch.sum(pdf_u_given_x, axis=0)
        return pdf_u_given_x

    def calculate_pdf_u(self, pdf_x):
        pdf_u = (self.pdf_u_given_x @ pdf_x) / torch.sum(self.pdf_u_given_x @ pdf_x)
        return pdf_u

    def calculate_delta_y(self, eps=1e-20):
        u_bnds = np.append(
            self.alphabet_u - self.config["delta_y"] / 2,
            self.alphabet_u[-1] + self.config["delta_y"] / 2,
        )
        y_bnds = self.nonlinear_fn(u_bnds)
        # Note that there is one negative value in difference - no clue
        self.delta_y = np.abs(np.ediff1d(y_bnds)) + eps

    def calculate_entropy_y(self, pdf_x, eps=1e-20):
        pdf_u = self.calculate_pdf_u(pdf_x)
        self.calculate_delta_y(eps)
        entropy_y = torch.sum(-pdf_u * torch.log((pdf_u) / self.delta_y))

        if torch.isnan(entropy_y):
            raise ValueError("Entropy is NaN")

        return entropy_y

    def calculate_entropy_y_given_x(self, pdf_x, eps):

        entropy_y_given_x = np.dot(
            torch.sum(
                -self.pdf_u_given_x
                * torch.log((self.pdf_u_given_x + eps) / (self.delta_y.reshape(-1, 1))),
                axis=0,
            ),
            pdf_x,
        )
        return entropy_y_given_x

    def capacity(self, pdf_x, eps=1e-20):
        entropy_y = self.calculate_entropy_y(pdf_x, eps)
        entropy_y_given_x = self.calculate_entropy_y_given_x(pdf_x, eps)
        # print(entropy_y, entropy_y_given_x)
        cap = entropy_y - entropy_y_given_x
        return cap
