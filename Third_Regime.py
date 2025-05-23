from Second_Regime import Second_Regime
from nonlinearity_utils import (
    get_nonlinear_fn,
    get_nonlinear_fn_numpy,
    get_derivative_of_inverse_nonlinearity,
    get_inverse_nonlinearity,
    Hardware_Nonlinear_and_Noise,
    real_quant,
)
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
        multiplying_factor=1,
    ):
        self.alphabet_x_re = alphabet_x.reshape(-1, 1)
        self.alphabet_x_im = alphabet_x_imag.reshape(1, -1) if config["complex"] else 0
        self.alphabet_y_re = alphabet_y.reshape(-1, 1)
        self.alphabet_y_im = alphabet_y_imag.reshape(1, -1) if config["complex"] else 0
        self.config = config
        self.power = power
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.tanh_factor = tanh_factor
        self.nonlinear_fn = get_nonlinear_fn(self.config, tanh_factor)
        self.get_x_and_y_alphabet()
        self.pdf_y_given_x = None
        self.q_pdf_y_given_x = None
        self.pdf_y_given_x_int = None
        self.get_x_and_y_alphabet()
        self.pdf_z1 = None
        self.alphabet_z1 = None
        self.pdf_y_given_x_and_x2 = None
        self.multiplying_factor = multiplying_factor
        if config["ADC"]:
            self.get_quant_param()

    def get_quant_param(self):
        n_levels = 2 ** self.config["bits"]

        # The maximum quantization location of y is decided accordingly without noise, what would be the mapping of Y
        max_y = min(
            self.nonlinear_fn(np.inf), self.nonlinear_fn(max(abs(self.alphabet_x)))
        )
        delta = (2 * max_y) / (n_levels)
        self.quant_locs = torch.linspace(
            -max_y + delta / 2, max_y - delta / 2, n_levels
        )

        if self.config["complex"]:
            self.quant_locs = self.quant_locs.reshape(
                -1, 1
            ) + 1j * self.quant_locs.reshape(1, -1)
            self.quant_locs = self.quant_locs.reshape(-1)

        self.distances = torch.abs(self.alphabet_y[:, None] - self.quant_locs[None, :])
        self.indices = torch.argmin(
            torch.abs(self.alphabet_y[:, None] - self.quant_locs[None, :]), dim=1
        )

    def get_z1_pdf_and_alphabet(self):
        if self.pdf_z1 is not None:
            return self.pdf_z1, self.alphabet_z1
        max_z1 = self.config["stop_sd"] * self.sigma_1

        delta_z1 = (self.alphabet_x_re[1] - self.alphabet_x_re[0])[0]
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

        self.pdf_z1 = pdf_z1
        self.alphabet_z1 = alphabet_z1
        return pdf_z1, alphabet_z1

    def get_pdf_y_given_x(self):
        if self.pdf_y_given_x is not None:
            # if self.config["ADC"]:
            #     return self.q_pdf_y_given_x
            # else:
            return self.pdf_y_given_x

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
                            self.alphabet_y.reshape(-1, 1) - alphabet_v.reshape(1, -1)
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
                            (self.alphabet_y.reshape(-1, 1) - alphabet_v.reshape(1, -1))
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

        # self.pdf_y_given_x = pdf_y_given_x
        if self.config["ADC"]:
            self.pdf_y_given_x = real_quant(
                self.quant_locs, self.indices, pdf_y_given_x
            )
            return self.pdf_y_given_x
        self.pdf_y_given_x = pdf_y_given_x
        return pdf_y_given_x

    def get_out_nonlinear(self, alphabet_u):
        if self.config["complex"]:
            r = self.nonlinear_fn(abs(alphabet_u))
            theta = torch.angle(alphabet_u)
            alphabet_v = r * torch.exp(1j * theta)
        else:
            alphabet_v = self.nonlinear_fn(alphabet_u)
        return alphabet_v

    def fix_with_multiplying(self, alphabet_x_RX2=None):
        if self.multiplying_factor == 1:
            return alphabet_x_RX2
        self.power = self.power / (10**self.multiplying_factor)
        self.alphabet_x_re = self.alphabet_x_re / (10 ** (self.multiplying_factor / 2))
        self.alphabet_y_re = self.alphabet_y_re / (10 ** (self.multiplying_factor / 2))
        self.alphabet_x_im = self.alphabet_x_im / (10 ** (self.multiplying_factor / 2))
        self.alphabet_y_im = self.alphabet_y_im / (10 ** (self.multiplying_factor / 2))
        self.sigma_1 = self.sigma_1 / (10 ** (self.multiplying_factor / 2))
        self.sigma_2 = self.sigma_2 / (10 ** (self.multiplying_factor / 2))

        self.config["iip3"] = self.config["iip3"] - 10 * self.multiplying_factor
        self.nonlinear_fn = get_nonlinear_fn(self.config)

        self.get_x_and_y_alphabet()
        if alphabet_x_RX2 is not None:
            alphabet_x_RX2 = alphabet_x_RX2 / (10 ** (self.multiplying_factor / 2))
            return alphabet_x_RX2

    def unfix_with_multiplying(self, alphabet_x_RX2=None):
        if self.multiplying_factor == 1:
            return alphabet_x_RX2

        self.power = self.power * (10**self.multiplying_factor)
        self.alphabet_x_re = self.alphabet_x_re * (10 ** (self.multiplying_factor / 2))
        self.alphabet_y_re = self.alphabet_y_re * (10 ** (self.multiplying_factor / 2))
        self.alphabet_x_im = self.alphabet_x_im * (10 ** (self.multiplying_factor / 2))
        self.alphabet_y_im = self.alphabet_y_im * (10 ** (self.multiplying_factor / 2))
        self.sigma_1 = self.sigma_1 * (10 ** (self.multiplying_factor / 2))
        self.sigma_2 = self.sigma_2 * (10 ** (self.multiplying_factor / 2))

        self.config["iip3"] = self.config["iip3"] + 10 * self.multiplying_factor
        self.nonlinear_fn = get_nonlinear_fn(self.config)
        self.get_x_and_y_alphabet()
        if self.config["ADC"]:
            self.get_quant_param()

        if alphabet_x_RX2 is not None:
            alphabet_x_RX2 = alphabet_x_RX2 * (10 ** (self.multiplying_factor / 2))
            return alphabet_x_RX2

    def new_capacity(self, pdf_x, eps=1e-20, pdf_y_given_x=None):
        self.fix_with_multiplying()

        if pdf_y_given_x is None:
            pdf_y_given_x = self.get_pdf_y_given_x()

        entropy_y_given_x = torch.sum(
            (pdf_y_given_x * torch.log(pdf_y_given_x + 1e-20)) @ pdf_x
        )
        pdf_y = pdf_y_given_x @ pdf_x
        pdf_y = pdf_y / (torch.sum(pdf_y) + 1e-30)
        cap = entropy_y_given_x - torch.sum(pdf_y * torch.log(pdf_y + 1e-20))

        self.unfix_with_multiplying()
        if cap.isnan():
            breakpoint()
        return cap

    def get_pdf_y_given_x1_and_x2(self, alphabet_x2, int_ratio):
        pdf_z1, alphabet_z1 = self.get_z1_pdf_and_alphabet()

        pdf_y_given_x_and_x2 = torch.zeros(
            self.alphabet_y.shape[0], self.alphabet_x.shape[0], len(alphabet_x2)
        )

        for ind, x in enumerate(self.alphabet_x):
            # U = X + Z_1 + aX_2
            alphabet_u = (
                alphabet_z1[None, None, :] + x + int_ratio * alphabet_x2[None, :, None]
            )
            # V = phi(U)

            alphabet_v = self.get_out_nonlinear(alphabet_u)
            if self.config["complex"]:
                pdf_y_given_x_x2_z1 = (
                    1
                    / (torch.pi * self.sigma_2**2)
                    * torch.exp(
                        -1
                        * abs(self.alphabet_y.reshape(-1, 1, 1) - alphabet_v) ** 2
                        / self.sigma_2**2
                    )
                )
            else:
                pdf_y_given_x_x2_z1 = (
                    1
                    / (torch.sqrt(torch.tensor([2 * torch.pi])) * self.sigma_2)
                    * torch.exp(
                        -0.5
                        * (abs(self.alphabet_y.reshape(-1, 1, 1) - alphabet_v) ** 2)
                        / self.sigma_2**2
                    )
                )

            pdf_y_given_x_x2_z1 = pdf_y_given_x_x2_z1 / (
                torch.sum(pdf_y_given_x_x2_z1, axis=0) + 1e-30
            )
            
            pdf_y_given_x_x2_z1 = pdf_y_given_x_x2_z1.to(dtype=pdf_z1.dtype)
            pdf_y_given_x_and_x2[:, ind, :] = pdf_y_given_x_x2_z1 @ pdf_z1

        pdf_y_given_x_and_x2 = pdf_y_given_x_and_x2 / (
            torch.sum(pdf_y_given_x_and_x2, axis=0) + 1e-30
        )

        if self.config["ADC"]:
            self.pdf_y_given_x_and_x2 = real_quant(
                self.quant_locs, self.indices, pdf_y_given_x_and_x2
            )
            return self.pdf_y_given_x_and_x2

        self.pdf_y_given_x_and_x2 = pdf_y_given_x_and_x2
        return self.pdf_y_given_x_and_x2

    def get_pdf_y_given_x_with_interference(self, pdf_x2, alphabet_x2, int_ratio):

        if self.pdf_y_given_x_and_x2 is None:
            self.get_pdf_y_given_x1_and_x2(alphabet_x2, int_ratio)

        pdf_y_given_x = self.pdf_y_given_x_and_x2 @ pdf_x2
        self.pdf_y_given_x_int = pdf_y_given_x
        return pdf_y_given_x

    def capacity_with_interference(
        self,
        pdf_x,
        pdf_x2,
        alphabet_x2,
        int_ratio,
    ):

        alphabet_x2 = self.fix_with_multiplying(alphabet_x2)

        if self.config["x2_fixed"] and self.pdf_y_given_x_int is not None:
            pdf_y_given_x = self.pdf_y_given_x_int
        else:

            pdf_y_given_x = self.get_pdf_y_given_x_with_interference(
                pdf_x2, alphabet_x2, int_ratio
            )

        entropy_y_given_x = torch.sum(
            (pdf_y_given_x * torch.log(pdf_y_given_x + 1e-20)) @ pdf_x
        )
        pdf_y = pdf_y_given_x @ pdf_x
        pdf_y = pdf_y / (torch.sum(pdf_y) + 1e-30)
        cap = entropy_y_given_x - torch.sum(pdf_y * torch.log(pdf_y + 1e-20))

        alphabet_x2 = self.unfix_with_multiplying(alphabet_x2)

        return cap

    def capacity_with_known_interference(
        self, pdf_x, pdf_x2, alphabet_x2, int_ratio, reg3_active=False
    ):

        # I(X1,Y1 | X2) = H(Y1|X2) - H(Y1|X1,X2)

        alphabet_x2 = self.fix_with_multiplying(alphabet_x2)

        if self.pdf_y_given_x_and_x2 is None:
            self.get_pdf_y_given_x1_and_x2(alphabet_x2, int_ratio)
        pdf_y_given_x2_and_x1 = torch.movedim(self.pdf_y_given_x_and_x2, 1, 2)

        pdf_y_given_x2 = pdf_y_given_x2_and_x1 @ pdf_x

        entropy_y_given_x2 = -torch.sum(
            (pdf_y_given_x2 * torch.log(pdf_y_given_x2 + 1e-20)) @ pdf_x2
        )

        entropy_y_given_x1_and_x2 = -torch.sum(
            ((pdf_y_given_x2_and_x1 * torch.log(pdf_y_given_x2_and_x1 + 1e-20)) @ pdf_x)
            @ pdf_x2
        )

        cap = entropy_y_given_x2 - entropy_y_given_x1_and_x2
        alphabet_x2 = self.unfix_with_multiplying(alphabet_x2)
        return cap

    def get_x_and_y_alphabet(self):
        if self.config["complex"]:
            self.alphabet_y = self.alphabet_y_re.reshape(
                -1, 1
            ) + 1j * self.alphabet_y_im.reshape(1, -1)
            self.alphabet_x = self.alphabet_x_re.reshape(
                -1, 1
            ) + 1j * self.alphabet_x_im.reshape(1, -1)
            self.alphabet_x = self.alphabet_x.reshape(-1)
            self.alphabet_y = self.alphabet_y.reshape(-1)
        else:
            self.alphabet_y = self.alphabet_y_re.reshape(-1)
            self.alphabet_x = self.alphabet_x_re.reshape(-1)

            # - Dead Funcs -#


# Just try real first
# and only hardware nonlinearity is considered
# def get_pdf_y_given_x_new(self):
#     if self.pdf_y_given_x is None:
#         pdf_y_given_x = torch.zeros(
#             (self.alphabet_y.shape[0], self.alphabet_x.shape[0])
#         )
#         delta = self.alphabet_x[1] - self.alphabet_x[0]

#         # max_nonlinear = self.nonlinear_fn(np.inf)
#         max_nonlinear = self.nonlinear_fn(
#             max(self.alphabet_x) + self.sigma_1 ** self.config["stop_sd"]
#         )

#         alphabet_v = torch.arange(
#             -max_nonlinear * 0.9999999,
#             max_nonlinear * 0.9999999,
#             delta,
#         )
#         d_inv_nonlinear = get_derivative_of_inverse_nonlinearity(
#             self.config, self.tanh_factor
#         )
#         inv_nonlinear = get_inverse_nonlinearity(self.config, self.tanh_factor)
#         gaus_dist1 = (
#             lambda alph, x: 1
#             / (torch.sqrt(torch.tensor([2 * torch.pi])) * self.sigma_1)
#             * torch.exp(-0.5 * (alph - x) ** 2 / self.sigma_1**2)
#         )

#         gaus_dist2 = (
#             lambda y, v: 1
#             / (torch.sqrt(torch.tensor([2 * torch.pi])) * self.sigma_2)
#             * torch.exp(
#                 -0.5 * (y.reshape(-1, 1) - v.reshape(1, -1)) ** 2 / self.sigma_2**2
#             )
#         )
#         # U = X + Z_1, V = phi(U)
#         # U|X ~ N(X, sigma_1^2)
#         # f_V|X = f_U|X (u) / |dphi/du|

#         for ind, x in enumerate(self.alphabet_x):

#             p_v_given_x = gaus_dist1(inv_nonlinear(alphabet_v), x) * abs(
#                 d_inv_nonlinear(alphabet_v)
#             )

#             p_v_given_x = p_v_given_x / (torch.sum(p_v_given_x, axis=0) + 1e-30)
#             breakpoint()
#             p_y_given_x = gaus_dist2(self.alphabet_y, alphabet_v) @ p_v_given_x
#             p_y_given_x = p_y_given_x / (torch.sum(p_y_given_x, axis=0) + 1e-30)
#             pdf_y_given_x[:, ind] = p_y_given_x
#         pdf_y_given_x = pdf_y_given_x / (torch.sum(pdf_y_given_x, axis=0) + 1e-30)
#         self.pdf_y_given_x = pdf_y_given_x
#     return self.pdf_y_given_x

# def get_pdfs_for_known_interference(self, pdf_x, pdf_x2, alphabet_x2, int_ratio):
#     pdf_z1, alphabet_z1 = self.get_z1_pdf_and_alphabet()
#     pdf_y_given_x2 = torch.zeros(len(self.alphabet_y), len(alphabet_x2))
#     pdf_y_given_x2_and_x1 = torch.zeros(
#         len(self.alphabet_y), len(alphabet_x2), len(self.alphabet_x)
#     )
#     for ind, x2 in enumerate(alphabet_x2):
#         mean_random_x2_x1_z1 = self.get_out_nonlinear(
#             int_ratio * x2 + self.alphabet_x[None, :, None] + alphabet_z1[None, None, :]
#         )
#         if self.config["complex"]:
#             pdf_y_given_x2_x1_z1 = (
#                 1
#                 / (torch.pi * self.sigma_2**2)
#                 * torch.exp(
#                     -1
#                     * abs(
#                         self.alphabet_y.reshape(
#                             -1,
#                             1,
#                             1,
#                         )
#                         - mean_random_x2_x1_z1
#                     )
#                     ** 2
#                     / self.sigma_2**2
#                 )
#             )
#         else:
#             pdf_y_given_x2_x1_z1 = (
#                 1
#                 / (torch.sqrt(torch.tensor([2 * torch.pi])) * self.sigma_2)
#                 * torch.exp(
#                     -0.5
#                     * (
#                         abs(
#                             self.alphabet_y.reshape(
#                                 -1,
#                                 1,
#                                 1,
#                             )
#                             - mean_random_x2_x1_z1
#                         )
#                         ** 2
#                     )
#                     / self.sigma_2**2
#                 )
#             )
#         pdf_y_given_x2_x1_z1 = pdf_y_given_x2_x1_z1 / (
#             torch.sum(pdf_y_given_x2_x1_z1, axis=0) + 1e-30
#         )

#         pdf_y_given_x2_and_x1_temp = pdf_y_given_x2_x1_z1 @ pdf_z1
#         pdf_y_given_x2_temp = pdf_y_given_x2_and_x1_temp @ pdf_x
#         pdf_y_given_x2[:, ind] = pdf_y_given_x2_temp / (
#             torch.sum(pdf_y_given_x2_temp, axis=0) + 1e-30
#         )
#         pdf_y_given_x2_and_x1[:, ind, :] = pdf_y_given_x2_and_x1_temp / (
#             torch.sum(pdf_y_given_x2_and_x1_temp, axis=0) + 1e-30
#         )
#     return pdf_y_given_x2_and_x1, pdf_y_given_x2


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
#     # np_nonlin =get_nonlinear_fn_numpy(self.config)
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
