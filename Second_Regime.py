import torch
from nonlinearity_utils import alphabet_fix_nonlinear
import numpy as np
import math
from nonlinearity_utils import get_nonlinear_fn


class Second_Regime:
    def __init__(self, alphabet_x, config, power):
        self.alphabet_x = alphabet_x
        self.config = config
        self.alphabet_u = self.calculate_u_points()  # U = X+Z_1
        self.pdf_u_given_x = self.calculate_pdf_u_given_x()
        self.power = power
        self.nonlinear_fn = get_nonlinear_fn(self.config)
        self.calculate_delta_y(eps=1e-35)

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
        pdf_u_given_x = pdf_u_given_x / (torch.sum(pdf_u_given_x, axis=0))
        return pdf_u_given_x

    def calculate_pdf_u(self, pdf_x):
        # breakpoint()
        pdf_u = (self.pdf_u_given_x @ pdf_x) / (torch.sum(self.pdf_u_given_x @ pdf_x))
        return pdf_u

    def calculate_delta_y(self, eps=1e-35):

        y_pts = self.nonlinear_fn(self.alphabet_u)

        y_pts_new, indices = torch.unique(y_pts, return_inverse=True)
        self.indices = indices
        # breakpoint()
        # self.pdf_u = torch.bincount(indices, weights=self.pdf_u)
        # temp = torch.zeros_like(self.pdf_u)
        # self.pdf_u = temp.scatter_reduce(
        #     0, indices, self.pdf_u, reduce="sum", include_self=False
        # )[: torch.max(indices) + 1]
        new_pdf_u_given_x = torch.zeros((len(y_pts_new), self.pdf_u_given_x.shape[1]))
        for i in range(self.pdf_u_given_x.shape[1]):
            new_pdf_u_given_x[:, i] = torch.bincount(
                indices, weights=self.pdf_u_given_x[:, i]
            )
        # breakpoint()
        self.pdf_u_given_x = new_pdf_u_given_x

        if y_pts_new.shape[0] > 1:
            y_bnds = torch.cat(
                (
                    y_pts_new - self.config["delta_y"] / 2,
                    torch.tensor([y_pts_new[-1] + self.config["delta_y"] / 2]),
                )
            )

            self.delta_y = torch.diff(y_bnds) + eps
        else:
            self.delta_y = eps

    def calculate_entropy_y(self, pdf_x, eps=1e-35):
        self.pdf_u = self.calculate_pdf_u(pdf_x)
        # self.calculate_delta_y(eps)

        # breakpoint()
        entropy_y = torch.sum(-self.pdf_u * torch.log((self.pdf_u) / self.delta_y))

        if torch.isnan(entropy_y):
            raise ValueError("Entropy is NaN")

        return entropy_y

    def calculate_entropy_y_given_x(self, pdf_x, eps):

        entropy_y_given_x = torch.dot(
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

    def capacity_like_ba(self, pdf_x):

        self.pdf_u = self.calculate_pdf_u(pdf_x)

        eps = 1e-35
        self.calculate_delta_y(eps)
        pdf_y_given_x = self.pdf_u_given_x
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

    # def set_alphabet_x(self, alphabet_x):
    #     self.alphabet_x = alphabet_x
    #     self.alphabet_u = self.calculate_u_points()
    #     self.pdf_u_given_x = self.calculate_pdf_u_given_x()
    #     self.calculate_delta_y(eps=1e-35)
