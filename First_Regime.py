import torch
from nonlinearity_utils import alphabet_fix_nonlinear
import numpy as np
from nonlinearity_utils import return_nonlinear_fn


class First_Regime:
    def __init__(self, alphabet_x, alphabet_y, config, power):
        self.alphabet_x = alphabet_x
        self.nonlinear_func = return_nonlinear_fn(config)
        self.alphabet_v = self.nonlinear_func(alphabet_x)  # V = phi(X+Z_1) when Z_1 = 0
        self.alphabet_y = alphabet_y  # Y = V + Z_2
        self.config = config
        self.pdf_y_given_v = self.calculate_pdf_y_given_v()
        self.power = power
        self.entropy_y_given_x = self.calculate_entropy_y_given_x()

    def set_alphabet_x(self, alphabet_x):
        self.alphabet_x = alphabet_x
        self.alphabet_v = self.nonlinear_func(alphabet_x)
        self.pdf_y_given_v = self.calculate_pdf_y_given_v()

    def set_alphabet_v(self, alphabet_v):
        self.alphabet_v = alphabet_v

    def calculate_pdf_y_given_v(self):
        pdf_y_given_v = (
            1
            / (torch.sqrt(torch.tensor([2 * torch.pi])) * self.config["sigma_2"])
            * torch.exp(
                -0.5
                * (
                    (self.alphabet_y.reshape(-1, 1) - self.alphabet_v.reshape(1, -1))
                    ** 2
                )
                / self.config["sigma_2"] ** 2
            )
        )
        pdf_y_given_v = pdf_y_given_v / (torch.sum(pdf_y_given_v, axis=0) )

        return pdf_y_given_v

    def calculate_entropy_y(self, pdf_x, eps=1e-20):

        # Projection is done with the main alphabet
        if len(self.alphabet_x) == 0:
            raise ValueError("Alphabet_x is empty")

        # pdf_x = pdf_u since one-one function
        pdf_y = (self.pdf_y_given_v @ pdf_x) / torch.sum(self.pdf_y_given_v @ pdf_x)
        entropy_y = torch.sum(-pdf_y * torch.log(pdf_y + eps)) + torch.log(
            torch.tensor([self.config["delta_y"]])
        )

        if torch.isnan(entropy_y):
            raise ValueError("Entropy is NaN")

        return entropy_y

    def calculate_entropy_y_given_x(self):
        entropy_y_given_x = 0.5 * torch.log(
            torch.tensor([2 * np.pi * np.e * self.config["sigma_2"] ** 2])
        )
        return entropy_y_given_x

    def capacity(self, pdf_x):
        entropy_y = self.calculate_entropy_y(pdf_x)
        cap = entropy_y - self.entropy_y_given_x
        return cap

    def capacity_like_ba(self, pdf_x):
        # breakpoint()
        pdf_y_given_x = self.pdf_y_given_v
        pdf_x_given_y = pdf_y_given_x * pdf_x
        pdf_x_given_y = torch.transpose(pdf_x_given_y, 0, 1) / torch.sum(
            pdf_x_given_y, axis=1
        )
        c = 0

        for i in range(len(self.alphabet_x)):
            if pdf_x[i] > 0:
                c += torch.sum(
                    pdf_x[i]
                    * pdf_y_given_x[:, i]
                    * torch.log(pdf_x_given_y[i, :] / pdf_x[i] + 1e-16)
                )
        c = c
        return c
