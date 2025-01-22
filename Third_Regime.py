from Second_Regime import Second_Regime
from nonlinearity_utils import return_nonlinear_fn, return_nonlinear_fn_numpy
from First_Regime import First_Regime
import torch
import time
import numpy as np


class Third_Regime:
    # Y = phi(X + Z_1) + Z_2, U = X+Z_1 and V = phi(U)
    def __init__(
        self,
        alphabet_x,
        alphabet_y,
        config,
        power,
        tanh_factor,
        sigma_1,
        sigma_2,
        alphabet_x_imag=0,
        alphabet_y_imag=0,
    ):
        self.alphabet_x_re = alphabet_x.reshape(-1)
        self.alphabet_x_im = alphabet_x_imag.reshape(-1)
        self.alphabet_y_re = alphabet_y.reshape(-1)
        self.alphabet_y_im = alphabet_y_imag.reshape(-1)
        self.config = config
        self.power = power
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.tanh_factor = tanh_factor
        self.nonlinear_fn = return_nonlinear_fn(self.config, tanh_factor)
        self.get_x_and_y_alphabet()
        self.pdf_y_given_x = None
        self.pdf_y_given_x_int = None
        self.get_x_and_y_alphabet()

    def get_z1_pdf_and_alphabet(self):
        max_z1 = self.config["stop_sd"] * self.sigma_1**2

        delta_z1 = self.alphabet_x_re[1] - self.alphabet_x_re[0]
        max_z1 = max_z1 + (delta_z1 - (max_z1 % delta_z1))
        # Z1 <- Gaussian noise with variance sigma_1^2

        if self.config["complex"]:
            # Z is also complex, Z = Z_re + 1j*Z_im
            alphabet_z1 = torch.arange(-max_z1, max_z1 + delta_z1 / 2, delta_z1)
            alphabet_z1 = alphabet_z1.reshape(1, -1) + 1j * alphabet_z1.reshape(-1, 1)

            alphabet_z1 = alphabet_z1.reshape(-1)
            pdf_z1 = (
                1
                / (torch.pi * self.sigma_1**2)
                * torch.exp(-1 * abs(alphabet_z1) ** 2 / self.sigma_1**2)
            )

        else:
            alphabet_z1 = torch.arange(-max_z1, max_z1 + delta_z1 / 2, delta_z1)
            pdf_z1 = (
                1
                / (torch.sqrt(torch.tensor([2 * torch.pi])) * self.sigma_1)
                * torch.exp(-0.5 * (alphabet_z1) ** 2 / self.sigma_1**2)
            )

        pdf_z1 = pdf_z1 / (torch.sum(pdf_z1) + 1e-30)
        return pdf_z1, alphabet_z1

    def get_pdf_y_given_x(self):
        if self.pdf_y_given_x is None:
            pdf_y_given_x = torch.zeros(
                (self.alphabet_y.shape[0], self.alphabet_x.shape[0])
            )
            # Z1 <- Gaussian noise with variance sigma_1^2

            pdf_z1, alphabet_z1 = self.get_z1_pdf_and_alphabet()

            for ind, x in enumerate(self.alphabet_x):
                # U = X + Z_1
                alphabet_u = alphabet_z1 + x
                # V = phi(U)

                alphabet_v = self.get_out_nonlinear(alphabet_u)
                if self.config["complex"]:
                    pdf_y_given_x_and_z1 = (
                        1
                        / (torch.pi * self.sigma_2**2)
                        * torch.exp(
                            -1
                            * abs(
                                self.alphabet_y.reshape(-1, 1)
                                - alphabet_v.reshape(1, -1)
                            )
                            ** 2
                            / self.sigma_2**2
                        )
                    )
                else:
                    pdf_y_given_x_and_z1 = (
                        1
                        / (torch.sqrt(torch.tensor([2 * torch.pi])) * self.sigma_2)
                        * torch.exp(
                            -0.5
                            * (
                                (
                                    self.alphabet_y.reshape(-1, 1)
                                    - alphabet_v.reshape(1, -1)
                                )
                                ** 2
                            )
                            / self.sigma_2**2
                        )
                    )
                pdf_y_given_x_and_z1 = pdf_y_given_x_and_z1 / (
                    torch.sum(pdf_y_given_x_and_z1, axis=0) + 1e-30
                )
                pdf_y_given_x[:, ind] = pdf_y_given_x_and_z1 @ pdf_z1
                pdf_y_given_x[:, ind] = pdf_y_given_x[:, ind] / (
                    torch.sum(pdf_y_given_x[:, ind]) + 1e-30
                )

            self.pdf_y_given_x = pdf_y_given_x
            return pdf_y_given_x
        else:
            return self.pdf_y_given_x

    def get_out_nonlinear(self, alphabet_u):
        if self.config["complex"]:
            r = self.nonlinear_fn(abs(alphabet_u))
            theta = torch.angle(alphabet_u)
            alphabet_v = r * torch.exp(1j * theta)
        else:
            alphabet_v = self.nonlinear_fn(alphabet_u)
        return alphabet_v

    def new_capacity(self, pdf_x, eps=1e-20):
        pdf_y_given_x = self.get_pdf_y_given_x()
        # pdf_y = torch.zeros_like(self.alphabet_y)
        # entropy_y_given_x = 0

        # for ind in range(len(self.alphabet_x)):
        #     plogp_y_given_x = pdf_y_given_x[:, ind] * torch.log(
        #         pdf_y_given_x[:, ind] + 1e-20
        #     )
        #     entropy_y_given_x += torch.sum(plogp_y_given_x * pdf_x[ind])
        #     pdf_y = pdf_y + pdf_y_given_x[:, ind] * pdf_x[ind]

        entropy_y_given_x = torch.sum(
            (pdf_y_given_x * torch.log(pdf_y_given_x + 1e-20)) @ pdf_x
        )
        pdf_y = pdf_y_given_x @ pdf_x
        pdf_y = pdf_y / (torch.sum(pdf_y) + 1e-30)
        cap = entropy_y_given_x - torch.sum(pdf_y * torch.log(pdf_y + 1e-20))

        return cap

    def get_pdf_y_given_x_with_interference(self, pdf_x2, alphabet_x2, int_ratio):
        pdf_y_given_x = torch.zeros(
            (self.alphabet_y.shape[0], self.alphabet_x.shape[0])
        )
        # Z1 <- Gaussian noise with variance sigma_1^2
        # X2 <- given by the user

        pdf_z1, alphabet_z1 = self.get_z1_pdf_and_alphabet()

        # Z_1 and X_2 are independent
        # Make sure that the pdf_x2 is normalized

        pdf_x2_and_z1 = pdf_x2[:, None] * pdf_z1[None, :]

        for ind, x in enumerate(self.alphabet_x):
            # U = X + Z_1 + aX_2
            alphabet_u = alphabet_z1[None, :] + x + int_ratio * alphabet_x2[:, None]
            # V = phi(U)

            alphabet_v = self.get_out_nonlinear(alphabet_u)
            if self.config["complex"]:
                pdf_y_given_x_z1_x2 = (
                    1
                    / (torch.pi * self.sigma_2**2)
                    * torch.exp(
                        -1
                        * abs(
                            self.alphabet_y.reshape(-1, 1) - alphabet_v.reshape(1, -1)
                        )
                        ** 2
                        / self.sigma_2**2
                    )
                )
            else:
                pdf_y_given_x_z1_x2 = (
                    1
                    / (torch.sqrt(torch.tensor([2 * torch.pi])) * self.sigma_2)
                    * torch.exp(
                        -0.5
                        * (
                            (self.alphabet_y.reshape(-1, 1) - alphabet_v.reshape(1, -1))
                            ** 2
                        )
                        / self.sigma_2**2
                    )
                )
            pdf_y_given_x_z1_x2 = pdf_y_given_x_z1_x2 / (
                torch.sum(pdf_y_given_x_z1_x2, axis=0) + 1e-30
            )
            pdf_y_given_x_temp = (pdf_y_given_x_z1_x2 @ pdf_x2_and_z1.reshape(-1, 1))[
                :, 0
            ]
            pdf_y_given_x[:, ind] = pdf_y_given_x_temp / (
                torch.sum(pdf_y_given_x_temp) + 1e-30
            )

        self.pdf_y_given_x_int = pdf_y_given_x
        return pdf_y_given_x

    def capacity_with_interference(
        self,
        pdf_x,
        pdf_x2,
        alphabet_x2,
        int_ratio,
    ):
        if self.config["x2_fixed"] and self.pdf_y_given_x_int is not None:
            pdf_y_given_x = self.pdf_y_given_x_int
        else:
            # pdf_y_given_x = self.get_pdf_y_given_x_with_interference_nofor(
            #     pdf_x2, alphabet_x2, int_ratio
            # )
            pdf_y_given_x = self.get_pdf_y_given_x_with_interference(
                pdf_x2, alphabet_x2, int_ratio
            )

        entropy_y_given_x = torch.sum(
            (pdf_y_given_x * torch.log(pdf_y_given_x + 1e-20)) @ pdf_x
        )
        pdf_y = pdf_y_given_x @ pdf_x
        pdf_y = pdf_y / (torch.sum(pdf_y) + 1e-30)
        cap = entropy_y_given_x - torch.sum(pdf_y * torch.log(pdf_y + 1e-20))

        return cap

    def get_pdfs_for_known_interference(self, pdf_x, pdf_x2, alphabet_x2, int_ratio):

        pdf_z1, alphabet_z1 = self.get_z1_pdf_and_alphabet()
        pdf_y_given_x2 = torch.zeros(len(self.alphabet_y), len(alphabet_x2))
        pdf_y_given_x2_and_x1 = torch.zeros(
            len(self.alphabet_y), len(alphabet_x2), len(self.alphabet_x)
        )
        for ind, x2 in enumerate(alphabet_x2):
            mean_random_x2_x1_z1 = self.get_out_nonlinear(
                int_ratio * x2
                + self.alphabet_x[None, :, None]
                + alphabet_z1[None, None, :]
            )
            if self.config["complex"]:
                pdf_y_given_x2_x1_z1 = (
                    1
                    / (torch.pi * self.sigma_2**2)
                    * torch.exp(
                        -1
                        * abs(
                            self.alphabet_y.reshape(
                                -1,
                                1,
                                1,
                            )
                            - mean_random_x2_x1_z1
                        )
                        ** 2
                        / self.sigma_2**2
                    )
                )
            else:
                pdf_y_given_x2_x1_z1 = (
                    1
                    / (torch.sqrt(torch.tensor([2 * torch.pi])) * self.sigma_2)
                    * torch.exp(
                        -0.5
                        * (
                            (
                                self.alphabet_y.reshape(
                                    -1,
                                    1,
                                    1,
                                )
                                - mean_random_x2_x1_z1
                            )
                            ** 2
                        )
                        / self.sigma_2**2
                    )
                )
            pdf_y_given_x2_x1_z1 = pdf_y_given_x2_x1_z1 / (
                torch.sum(pdf_y_given_x2_x1_z1, axis=0) + 1e-30
            )

            pdf_y_given_x2_and_x1_temp = pdf_y_given_x2_x1_z1 @ pdf_z1
            pdf_y_given_x2_temp = pdf_y_given_x2_and_x1_temp @ pdf_x
            pdf_y_given_x2[:, ind] = pdf_y_given_x2_temp / (
                torch.sum(pdf_y_given_x2_temp, axis=0) + 1e-30
            )
            pdf_y_given_x2_and_x1[:, ind, :] = pdf_y_given_x2_and_x1_temp / (
                torch.sum(pdf_y_given_x2_and_x1_temp, axis=0) + 1e-30
            )
        return pdf_y_given_x2_and_x1, pdf_y_given_x2

    def capacity_with_known_interference(self, pdf_x, pdf_x2, alphabet_x2, int_ratio):
        pdf_y_given_x2_and_x1, pdf_y_given_x2 = self.get_pdfs_for_known_interference(
            pdf_x, pdf_x2, alphabet_x2, int_ratio
        )
        # pdf_y_given_x2_and_x1, pdf_y_given_x2 = (
        #     self.get_pdfs_for_known_interference_nofor(pdf_x, pdf_x2, alphabet_x2, int_ratio)
        # )
        # I(X1,Y1 | X2) = H(Y1|X2) - H(Y1|X1,X2)

        entropy_y_given_x2 = -torch.sum(
            (pdf_y_given_x2 * torch.log(pdf_y_given_x2 + 1e-20)) @ pdf_x2
        )

        entropy_y_given_x1_and_x2 = -torch.sum(
            ((pdf_y_given_x2_and_x1 * torch.log(pdf_y_given_x2_and_x1 + 1e-20)) @ pdf_x)
            @ pdf_x2
        )

        # sum_x = 0
        # for i in range(len(self.alphabet_x)):
        #     for j in range(len(alphabet_x2)):
        #         sum_x = sum_x + pdf_x[i] * pdf_x2[j] * torch.sum(
        #             pdf_y_given_x2_and_x1[:, j, i]
        #             * torch.log(1 / (pdf_y_given_x2_and_x1[:, j, i] + 1e-30) + 1e-30)
        #         )

        cap = entropy_y_given_x2 - entropy_y_given_x1_and_x2

        return cap

    def get_x_and_y_alphabet(self):
        if self.config["complex"]:
            self.alphabet_y = self.alphabet_y_re + 1j * self.alphabet_y_im
            self.alphabet_x = self.alphabet_x_re + 1j * self.alphabet_x_im
            self.alphabet_x = self.alphabet_x.reshape(-1)
            self.alphabet_y = self.alphabet_y.reshape(-1)
        else:
            self.alphabet_y = self.alphabet_y_re
            self.alphabet_x = self.alphabet_x_re

            # - Dead Funcs -#

    # def get_pdf_y_given_x_with_interference_nofor(self, pdf_x2, alphabet_x2, int_ratio):

    #     max_z1 = self.config["stop_sd"] * self.config["sigma_11"] ** 2
    #     delta_z1 = self.alphabet_x[1] - self.alphabet_x[0]
    #     max_z1 = max_z1 + (delta_z1 - (max_z1 % delta_z1))
    #     alphabet_z1 = torch.arange(-max_z1, max_z1 + delta_z1 / 2, delta_z1)

    #     pdf_z1 = (
    #         1
    #         / (torch.sqrt(torch.tensor([2 * torch.pi])) * self.config["sigma_11"])
    #         * torch.exp(-0.5 * (alphabet_z1) ** 2 / self.config["sigma_11"] ** 2)
    #     )
    #     pdf_z1 = pdf_z1 / (torch.sum(pdf_z1) + 1e-30)

    #     # alphabet_z1 = np.arange(-max_z1, max_z1 + delta_z1 / 2, delta_z1)
    #     # pdf_z1 = (
    #     #     1
    #     #     / (np.sqrt(2 * np.pi) * self.config["sigma_11"])
    #     #     * np.exp(-0.5 * (alphabet_z1) ** 2 / self.config["sigma_11"] ** 2)
    #     # )
    #     # pdf_z1 = pdf_z1 / (np.sum(pdf_z1) + 1e-30)
    #     # np_nonlin = return_nonlinear_fn_numpy(self.config)
    #     mean_random_x1_x2_z1 = self.nonlinear_fn(
    #         self.alphabet_x[None, :, None, None]
    #         + int_ratio * np.array(alphabet_x2[None, None, :, None])
    #         + alphabet_z1[None, None, None, :]
    #     )

    #     pdf_y_given_x1_x2_z1 = (
    #         1
    #         / (torch.sqrt(torch.tensor([2 * torch.pi])) * self.config["sigma_12"])
    #         * torch.exp(
    #             -0.5
    #             * ((self.alphabet_y.reshape(-1, 1, 1, 1) - mean_random_x1_x2_z1) ** 2)
    #             / self.config["sigma_12"] ** 2
    #         )
    #     )
    #     # pdf_y_given_x1_x2_z1 = (
    #     #     1
    #     #     / (np.sqrt(2 * np.pi) * self.config["sigma_12"])
    #     #     * np.exp(
    #     #         -0.5
    #     #         * (
    #     #             (
    #     #                 np.array(self.alphabet_y).reshape(-1, 1, 1, 1)
    #     #                 - mean_random_x1_x2_z1
    #     #             )
    #     #             ** 2
    #     #         )
    #     #         / self.config["sigma_12"] ** 2
    #     #     )
    #     # )

    #     pdf_y_given_x1_x2 = pdf_y_given_x1_x2_z1 @ pdf_z1
    #     pdf_y_given_x1 = pdf_y_given_x1_x2 @ pdf_x2
    #     # pdf_y_given_x1 = torch.tensor(pdf_y_given_x1_x2.astype(np.float32)) @ pdf_x2
    #     pdf_y_given_x1 = pdf_y_given_x1 / (torch.sum(pdf_y_given_x1, axis=0) + 1e-30)
    #     self.pdf_y_given_x_int = pdf_y_given_x1
    #     return pdf_y_given_x1

    # def get_pdfs_for_known_interference_nofor(
    #     self, pdf_x, pdf_x2, alphabet_x2, int_ratio
    # ):

    #     max_z1 = self.config["stop_sd"] * self.config["sigma_11"] ** 2
    #     delta_z1 = self.alphabet_x[1] - self.alphabet_x[0]
    #     max_z1 = max_z1 + (delta_z1 - (max_z1 % delta_z1))
    #     alphabet_z1 = torch.arange(-max_z1, max_z1 + delta_z1 / 2, delta_z1)
    #     pdf_z1 = (
    #         1
    #         / (torch.sqrt(torch.tensor([2 * torch.pi])) * self.config["sigma_11"])
    #         * torch.exp(-0.5 * (alphabet_z1) ** 2 / self.config["sigma_11"] ** 2)
    #     )
    #     pdf_z1 = pdf_z1 / (torch.sum(pdf_z1) + 1e-30)

    #     mean_random_x2_x1_z1 = self.nonlinear_fn(
    #         int_ratio * alphabet_x2[None, :, None, None]
    #         + self.alphabet_x[None, None, :, None]
    #         + alphabet_z1[None, None, None, :]
    #     )
    #     pdf_y_given_x2_x1_z1 = (
    #         1
    #         / (torch.sqrt(torch.tensor([2 * torch.pi])) * self.config["sigma_12"])
    #         * torch.exp(
    #             -0.5
    #             * ((self.alphabet_y.reshape(-1, 1, 1, 1) - mean_random_x2_x1_z1) ** 2)
    #             / self.config["sigma_12"] ** 2
    #         )
    #     )
    #     pdf_y_given_x2_x1_z1 = pdf_y_given_x2_x1_z1 / (
    #         torch.sum(pdf_y_given_x2_x1_z1, axis=0) + 1e-30
    #     )

    #     pdf_y_given_x2_and_x1 = pdf_y_given_x2_x1_z1 @ pdf_z1
    #     pdf_y_given_x2 = pdf_y_given_x2_and_x1 @ pdf_x
    #     pdf_y_given_x2 = pdf_y_given_x2 / (torch.sum(pdf_y_given_x2, axis=0) + 1e-30)
    #     pdf_y_given_x2_and_x1 = pdf_y_given_x2_and_x1 / (
    #         torch.sum(pdf_y_given_x2_and_x1, axis=0) + 1e-30
    #     )
    #     return pdf_y_given_x2_and_x1, pdf_y_given_x2

    # def set_alphabet_x(self, alphabet_x):
    #     self.alphabet_x = alphabet_x
    #     self.s_regime = Second_Regime(alphabet_x, self.config, self.power)
    #     self.alphabet_v = self.nonlinear_fn(self.s_regime.alphabet_u)
    #     self.pdf_u_given_x = self.s_regime.calculate_pdf_u_given_x()
    #     self.f_regime = First_Regime(
    #         alphabet_x, self.alphabet_y, self.config, self.power
    #     )
    #     self.f_regime.set_alphabet_v(
    #         self.alphabet_v
    #     )  # since Z_1 is not 0, we define new U and V
    #     self.pdf_y_given_v = self.f_regime.calculate_pdf_y_given_v()
    #     self.pdf_y_given_u = self.pdf_y_given_v
    #     self.update_after_indices()
    # def update_after_indices(self):
    #     new_pdf_y_given_v = torch.zeros(
    #         (self.pdf_y_given_v.shape[0], max(self.s_regime.indices) + 1)
    #     )
    #     for i in range(self.pdf_y_given_v.shape[0]):
    #         new_pdf_y_given_v[i, :] = torch.bincount(
    #             self.s_regime.indices, weights=self.pdf_y_given_v[i, :]
    #         )
    #     self.pdf_y_given_v = new_pdf_y_given_v
    #     self.pdf_y_given_u = self.pdf_y_given_v

    #     new_pdf_u_given_x = torch.zeros(
    #         (max(self.s_regime.indices) + 1, self.pdf_u_given_x.shape[1])
    #     )
    #     for i in range(self.pdf_u_given_x.shape[1]):
    #         new_pdf_u_given_x[:, i] = torch.bincount(
    #             self.s_regime.indices, weights=self.pdf_u_given_x[:, i]
    #         )
    #     self.pdf_u_given_x = new_pdf_u_given_x

    # def calculate_entropy_y(self, pdf_x, eps):
    #     pdf_v = self.s_regime.calculate_pdf_u(pdf_x)
    #     pdf_y = (self.pdf_y_given_v @ pdf_v) / torch.sum(self.pdf_y_given_v @ pdf_v)
    #     entropy_y = torch.sum(-pdf_y * torch.log(pdf_y + eps)) + torch.log(
    #         torch.tensor([self.config["delta_y"]])
    #     )
    #     return entropy_y

    # def calculate_pdf_y_given_x(self, pdf_x):
    #     # f_y_given_x = f_x_and_y / f_x and f_x_and_y = sum over u (f_x_given_u * f_y_given_u* f_u)
    #     pdf_u = self.s_regime.calculate_pdf_u(pdf_x)

    #     pdf_x_given_u = torch.transpose(self.pdf_u_given_x, 0, 1) / (pdf_u)
    #     pdf_x_and_y_given_u = pdf_x_given_u[:, None, :] * self.pdf_y_given_u[None, :, :]
    #     pdf_x_and_y = pdf_x_and_y_given_u @ pdf_u
    #     pdf_y_given_x = torch.transpose(pdf_x_and_y, 0, 1)
    #     pdf_y_given_x = pdf_y_given_x / (torch.sum(pdf_y_given_x, axis=0))
    #     return pdf_y_given_x

    # def calculate_entropy_y_given_x(self, pdf_x, eps):
    #     pdf_y_given_x = self.calculate_pdf_y_given_x(pdf_x)

    #     entropy_y_given_x = (
    #         torch.sum(
    #             -pdf_y_given_x
    #             * torch.log((pdf_y_given_x + eps) / (self.config["delta_y"])),
    #             axis=0,
    #         )
    #         @ pdf_x
    #     )

    #     return entropy_y_given_x

    # def capacity(self, pdf_x, eps=1e-20):
    #     entropy_y = self.calculate_entropy_y(pdf_x, eps)
    #     entropy_y_given_x = self.calculate_entropy_y_given_x(pdf_x, eps)
    #     cap = entropy_y - entropy_y_given_x
    #     return cap

    # def capacity_like_ba(self, pdf_x):
    #     if self.config["sigma_1"] <= 1e-5:
    #         return self.f_regime.capacity_like_ba(pdf_x)
    #     if self.config["sigma_2"] <= 1e-5:
    #         return self.s_regime.capacity_like_ba(pdf_x)

    #     pdf_y_given_x = self.calculate_pdf_y_given_x(pdf_x)
    #     pdf_x_given_y = pdf_y_given_x * pdf_x
    #     pdf_x_given_y = torch.transpose(pdf_x_given_y, 0, 1) / (
    #         torch.sum(pdf_x_given_y, axis=1)
    #     )
    #     c = 0

    #     for i in range(len(self.alphabet_x)):
    #         if pdf_x[i] > 0:
    #             c += torch.sum(
    #                 pdf_x[i]
    #                 * pdf_y_given_x[:, i]
    #                 * torch.log(pdf_x_given_y[i, :] / pdf_x[i] + 1e-16)
    #             )
    #     c = c
    #     return c
