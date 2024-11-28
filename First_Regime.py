import torch
from nonlinearity_utils import alphabet_fix_nonlinear
import numpy as np
from nonlinearity_utils import return_nonlinear_fn
import math
from Second_Regime import Second_Regime
from scipy.signal import fftconvolve
import torchaudio


class First_Regime:
    def __init__(
        self, alphabet_x, alphabet_y, config, power=None, interference_active=False
    ):
        self.alphabet_x = alphabet_x
        self.nonlinear_func = return_nonlinear_fn(config)
        self.alphabet_v = self.nonlinear_func(alphabet_x)  # V = phi(X+Z_1) when Z_1 = 0
        self.alphabet_y = alphabet_y  # Y = V + Z_2
        self.config = config
        self.pdf_y_given_v = self.calculate_pdf_y_given_v()
        self.power = power
        self.entropy_y_given_x = self.calculate_entropy_y_given_x()
        self.interference_active = interference_active
        self.delta = self.alphabet_x[1] - self.alphabet_x[0]

    def set_alphabet_x(self, alphabet_x):
        self.alphabet_x = alphabet_x
        self.alphabet_v = self.nonlinear_func(alphabet_x)
        self.pdf_y_given_v = self.calculate_pdf_y_given_v()

    def set_alphabet_v(self, alphabet_v):
        self.alphabet_v = alphabet_v

    def calculate_pdf_y_given_v(self, alphabet_v=None):
        if alphabet_v is None:
            alphabet_v = self.alphabet_v

        pdf_y_given_v = (
            1
            / (torch.sqrt(torch.tensor([2 * torch.pi])) * self.config["sigma_2"])
            * torch.exp(
                -0.5
                * ((self.alphabet_y.reshape(-1, 1) - alphabet_v.reshape(1, -1)) ** 2)
                / self.config["sigma_2"] ** 2
            )
        )
        pdf_y_given_v = pdf_y_given_v / (torch.sum(pdf_y_given_v, axis=0) + 1e-30)

        return pdf_y_given_v

    def calculate_entropy_y(self, pdf_x, eps=1e-20):

        # Projection is done with the main alphabet
        if len(self.alphabet_x) == 0:
            raise ValueError("Alphabet_x is empty")

        # pdf_x = pdf_u since one-one function
        pdf_y = (self.pdf_y_given_v @ pdf_x) / torch.sum(self.pdf_y_given_v @ pdf_x)
        entropy_y = torch.sum(-pdf_y * torch.log(pdf_y + eps)) + torch.log(self.delta)

        if torch.isnan(entropy_y):
            raise ValueError("Entropy is NaN")

        return entropy_y

    def calculate_entropy_y_given_x(self):
        entropy_y_given_x = 0.5 * torch.log(
            torch.tensor([2 * np.pi * np.e * self.config["sigma_2"] ** 2])
        )
        return entropy_y_given_x

    # Note: Not using this currently
    def capacity(self, pdf_x):
        if not self.interference_active:
            entropy_y = self.calculate_entropy_y(pdf_x)
            cap = entropy_y - self.entropy_y_given_x
        else:
            # FIXME: For not gaussian input distributions
            # self.pdf_u = (
            #     1
            #     / (
            #         torch.sqrt(
            #             torch.tensor(
            #                 [
            #                     2
            #                     * torch.pi
            #                     * (1 + self.config["int_ratio"] ** 2)
            #                     * self.power
            #                 ]
            #             )
            #         )
            #     )
            #     * torch.exp(
            #         -0.5
            #         * ((self.alphabet_u) ** 2)
            #         / ((1 + self.config["int_ratio"] ** 2) * self.power)
            #     ).float()
            # )

            # self.pdf_u = np.convolve(self.pdf_int, pdf_x)#This gives one extra
            # breakpoint()
            # self.pdf_u = fftconvolve(self.pdf_int, pdf_x, mode="full") - Does not work with gradients
            self.pdf_u = torchaudio.functional.fftconvolve(
                self.pdf_int, pdf_x, mode="full"
            )
            if len(self.pdf_u) != len(self.alphabet_u):
                breakpoint()

            self.pdf_u = (
                torch.tensor(self.pdf_u) / torch.sum(torch.tensor(self.pdf_u))
            ).to(torch.float32)
            if torch.sum(self.pdf_u < 0) > 0:
                self.pdf_u = self.pdf_u - torch.min(self.pdf_u[self.pdf_u < 0]) + 1e-20
                # breakpoint()
            entropy_y = self.calculate_entropy_y(self.pdf_u)
            # TODO: entropy y given x degisiyor
            entropy_y_given_x = self.calculate_entropy_y_given_x_interference(pdf_x)
            cap = entropy_y - entropy_y_given_x
            if cap.isnan():
                breakpoint()
        return cap

    def capacity_like_ba(self, pdf_x):
        # breakpoint()
        pdf_y_given_x = self.pdf_y_given_v
        pdf_x_given_y = pdf_y_given_x * pdf_x
        pdf_x_given_y = torch.transpose(pdf_x_given_y, 0, 1) / torch.sum(
            pdf_x_given_y, axis=1
        )
        c = 0
        # breakpoint()
        for i in range(len(self.alphabet_x)):
            if pdf_x[i] > 0:
                c += torch.sum(
                    pdf_x[i]
                    * pdf_y_given_x[:, i]
                    * torch.log(pdf_x_given_y[i, :] / pdf_x[i] + 1e-16)
                )
        c = c
        return c

    def capacity_like_ba_for_alphabet_x(self, alphabet_x):
        pass

    def set_interference_active(self, int_alphabet_x, int_pdf_x):
        # breakpoint()
        # max_u = max(self.alphabet_x) + max(int_alphabet_x) * self.config["int_ratio"]
        # fix to be in delta_y-- but currently int_ratio is chosen to be integer multiple of delta_y
        # max_u = max_u + self.config["delta_y"] - (max_u % self.config["delta_y"])
        # self.alphabet_u = torch.arange(-max_u, max_u, self.config["delta_y"])

        # sample_u = math.ceil(2 * max_u / self.config["delta_y"]) + 1
        # self.alphabet_u = torch.linspace(-max_u, max_u, sample_u)
        # self.int_alphabet_x = int_alphabet_x
        # self.alphabet_v = self.nonlinear_func(self.alphabet_u)
        # self.pdf_y_given_v = self.calculate_pdf_y_given_v()
        self.interference_active = True

        # This is the probability distribution of the aX2 so a will be in the pmf calculation
        self.pdf_int, self.int_alphabet_x = self.update_pdf_int_for_ratio(
            int_pdf_x, int_alphabet_x, self.config["int_ratio"]
        )
        max_u = max(self.alphabet_x) + max(self.int_alphabet_x)
        self.alphabet_u = torch.arange(-max_u, max_u + self.delta / 2, self.delta)
        self.alphabet_v = self.nonlinear_func(self.alphabet_u)
        self.pdf_y_given_v = self.calculate_pdf_y_given_v()
        # breakpoint()

    def update_pdf_int_for_ratio(self, int_pdf_x, int_alphabet_x, ratio):
        # Y = aX2
        # pdf_y = pdf_x(x/a) * |1/a|
        y_points = int_alphabet_x * ratio
        pdf_y = torch.zeros_like(int_alphabet_x)
        for ind, x_2 in enumerate(int_alphabet_x):
            ind_x = torch.where(abs(self.alphabet_x - x_2 / ratio) < 1e-5)[0]
            if ind_x.size != 0:
                pdf_y[ind] = int_pdf_x[ind_x]
            else:
                pdf_y[ind] = 0
        # breakpoint()
        pdf_y = pdf_y / torch.sum(pdf_y).to(torch.float32)
        return pdf_y, y_points

    def calculate_entropy_y_given_x_interference(self, pdf_x):

        pdf_y_given_x = self.calculate_pdf_y_given_x_interference()

        entropy_y_given_x = (
            torch.sum(
                -pdf_y_given_x * torch.log((pdf_y_given_x + 1e-20) / (self.delta)),
                axis=0,
            )
            @ pdf_x
        )
        if entropy_y_given_x.isnan():
            breakpoint()

        return entropy_y_given_x

    def calculate_pdf_y_given_x_interference(self):
        # U  = X_1 + aX_2
        # pdf_u = np.convolve(self.pdf_int, pdf_x)
        pdf_u_given_x = torch.zeros((len(self.alphabet_u), len(self.alphabet_x)))
        for i in range(len(self.alphabet_x)):
            u_temp = self.alphabet_x[i] + self.int_alphabet_x
            # breakpoint()
            # -Following does not work exactly
            # ind_temp = np.digitize(u_temp, self.alphabet_u, right=True)

            # --Shift can be found by finding the first and last indices
            f_ind = np.where(abs(u_temp[0] - self.alphabet_u) < self.delta / 2)[0]
            l_ind = np.where(abs(u_temp[-1] - self.alphabet_u) < self.delta / 2)[0]
            if l_ind.size == 0:
                l_ind = [len(self.alphabet_u)]
                breakpoint()
            # breakpoint()
            # X gives the shift in the probabilities corresponding to that u
            pdf_u_given_x[f_ind[0] : l_ind[0] + 1, i] = self.pdf_int[
                : l_ind[0] - f_ind[0] + 1
            ]
        pdf_u_given_x = pdf_u_given_x / (torch.sum(pdf_u_given_x, axis=0)).to(
            torch.float32
        )
        # Why convolve gives 0 pdf output ???
        pdf_x_given_u = torch.transpose(pdf_u_given_x, 0, 1) / (self.pdf_u + 1e-20)
        pdf_y_given_u = self.calculate_pdf_y_given_v(self.alphabet_v)
        pdf_x_and_y_given_u = pdf_x_given_u[:, None, :] * pdf_y_given_u[None, :, :]
        pdf_x_and_y = pdf_x_and_y_given_u @ self.pdf_u
        pdf_y_given_x = torch.transpose(pdf_x_and_y, 0, 1)
        if torch.sum(pdf_y_given_x < 0) > 0:
            breakpoint()
        pdf_y_given_x = pdf_y_given_x / (torch.sum(pdf_y_given_x, axis=0))
        return pdf_y_given_x

    def set_pdf_u(self, pdf_x):
        self.pdf_u = fftconvolve(self.pdf_int, pdf_x, mode="full")
        if len(self.pdf_u) != len(self.alphabet_u):
            breakpoint()

        self.pdf_u = (
            torch.tensor(self.pdf_u) / torch.sum(torch.tensor(self.pdf_u))
        ).to(torch.float32)

    def capacity_with_interference(self, pdf_x_RX1, pdf_x_RX2, alphabet_x_RX2):

        # max_y = (
        #     self.nonlinear_func(max(self.alphabet_x) + max(alphabet_x_RX2))
        #     + self.config["sigma_2"] * self.config["stop_sd"]
        # )
        # max_y = max_y + (self.delta - (max_y % self.delta))

        # alphabet_y = torch.arange(-max_y, max_y + self.delta / 2, self.delta)
        # pdf_y_given_x1_r = torch.zeros((len(alphabet_y), len(self.alphabet_x)))
        pdf_y = torch.zeros_like(self.alphabet_y)
        entropy_y_given_x = 0
        # breakpoint()
        for ind, x in enumerate(self.alphabet_x):
            # U = X1 + aX2, a = 1 #FIXME: a is fixed to 1
            alphabet_u = self.config["int_ratio"] * alphabet_x_RX2 + x
            alphabet_v = self.nonlinear_func(alphabet_u)
            pdf_y_given_x1_and_x2 = (
                1
                / (torch.sqrt(torch.tensor([2 * torch.pi])) * self.config["sigma_2"])
                * torch.exp(
                    -0.5
                    * (
                        (self.alphabet_y.reshape(-1, 1) - alphabet_v.reshape(1, -1))
                        ** 2
                    )
                    / self.config["sigma_2"] ** 2
                )
            )
            pdf_y_given_x1_and_x2 = pdf_y_given_x1_and_x2 / (
                torch.sum(pdf_y_given_x1_and_x2, axis=0) + 1e-30
            )
            # breakpoint()
            # pdf_y_given_x1_r[:, ind] = pdf_y_given_x1_and_x2 @ pdf_x_RX2

            pdf_y_given_x1 = (
                pdf_y_given_x1_and_x2 @ pdf_x_RX2 / self.config["int_ratio"]
            )
            pdfy_given_x1 = pdf_y_given_x1 / (torch.sum(pdf_y_given_x1, axis=0) + 1e-30)
            plogp_y_given_x = pdfy_given_x1 * torch.log(pdfy_given_x1 + 1e-20)
            entropy_y_given_x += torch.sum(plogp_y_given_x * pdf_x_RX1[ind])

            pdf_y = pdf_y + pdf_y_given_x1 * pdf_x_RX1[ind]
            # breakpoint()
        # pdf_y_given_x1_r = pdf_y_given_x1_r / (
        #     torch.sum(pdf_y_given_x1_r, axis=0) + 1e-30
        # )
        pdf_y = pdf_y / (torch.sum(pdf_y) + 1e-30)

        # plogp_y_given_x = pdf_y_given_x1_r * torch.log(pdf_y_given_x1_r + 1e-20)
        # px_plogp_y_given_x = plogp_y_given_x @ pdf_x_RX1
        # cap = torch.sum(px_plogp_y_given_x) - torch.sum(
        #     pdf_y * torch.log(pdf_y + 1e-20)
        # )
        cap = entropy_y_given_x - torch.sum(pdf_y * torch.log(pdf_y + 1e-20))
        # breakpoint()

        return cap

    def new_capacity(self, pdf_x):
        pdf_y_given_x = (
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

        pdf_y_given_x = pdf_y_given_x / (torch.sum(pdf_y_given_x, axis=0) + 1e-30)
        py_x_logpy_x = pdf_y_given_x * torch.log(pdf_y_given_x + 1e-20)
        px_py_x_logpy_x = py_x_logpy_x @ pdf_x
        f_term = torch.sum(px_py_x_logpy_x)
        py = pdf_y_given_x @ pdf_x
        s_term = torch.sum(py * torch.log(py + 1e-20))
        return f_term - s_term

    