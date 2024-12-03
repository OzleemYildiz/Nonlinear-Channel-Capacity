from Second_Regime import Second_Regime
from nonlinearity_utils import return_nonlinear_fn
from First_Regime import First_Regime
import torch


class Third_Regime:
    # Y = phi(X + Z_1) + Z_2, U = X+Z_1 and V = phi(U)
    def __init__(self, alphabet_x, alphabet_y, config, power):
        self.alphabet_x = alphabet_x
        self.config = config
        self.power = power
        self.nonlinear_fn = return_nonlinear_fn(self.config)
        self.alphabet_y = alphabet_y

        # self.s_regime = Second_Regime(alphabet_x, config, power)
        # self.alphabet_v = self.nonlinear_fn(self.s_regime.alphabet_u)
        # self.pdf_u_given_x = self.s_regime.calculate_pdf_u_given_x()

        # self.f_regime = First_Regime(alphabet_x, alphabet_y, config, power)
        # self.f_regime.set_alphabet_v(
        #     self.alphabet_v
        # )  # since Z_1 is not 0, we define new U and V
        # self.pdf_y_given_v = self.f_regime.calculate_pdf_y_given_v()
        # self.pdf_y_given_u = self.pdf_y_given_v
        # self.update_after_indices()
        self.pdf_y_given_x = None
        self.pdf_y_given_x_int = None

    def set_alphabet_x(self, alphabet_x):
        self.alphabet_x = alphabet_x
        self.s_regime = Second_Regime(alphabet_x, self.config, self.power)
        self.alphabet_v = self.nonlinear_fn(self.s_regime.alphabet_u)
        self.pdf_u_given_x = self.s_regime.calculate_pdf_u_given_x()
        self.f_regime = First_Regime(
            alphabet_x, self.alphabet_y, self.config, self.power
        )
        self.f_regime.set_alphabet_v(
            self.alphabet_v
        )  # since Z_1 is not 0, we define new U and V
        self.pdf_y_given_v = self.f_regime.calculate_pdf_y_given_v()
        self.pdf_y_given_u = self.pdf_y_given_v
        self.update_after_indices()

    def update_after_indices(self):
        new_pdf_y_given_v = torch.zeros(
            (self.pdf_y_given_v.shape[0], max(self.s_regime.indices) + 1)
        )
        for i in range(self.pdf_y_given_v.shape[0]):
            new_pdf_y_given_v[i, :] = torch.bincount(
                self.s_regime.indices, weights=self.pdf_y_given_v[i, :]
            )
        self.pdf_y_given_v = new_pdf_y_given_v
        self.pdf_y_given_u = self.pdf_y_given_v

        new_pdf_u_given_x = torch.zeros(
            (max(self.s_regime.indices) + 1, self.pdf_u_given_x.shape[1])
        )
        for i in range(self.pdf_u_given_x.shape[1]):
            new_pdf_u_given_x[:, i] = torch.bincount(
                self.s_regime.indices, weights=self.pdf_u_given_x[:, i]
            )
        self.pdf_u_given_x = new_pdf_u_given_x

    def calculate_entropy_y(self, pdf_x, eps):
        pdf_v = self.s_regime.calculate_pdf_u(pdf_x)
        pdf_y = (self.pdf_y_given_v @ pdf_v) / torch.sum(self.pdf_y_given_v @ pdf_v)
        entropy_y = torch.sum(-pdf_y * torch.log(pdf_y + eps)) + torch.log(
            torch.tensor([self.config["delta_y"]])
        )
        return entropy_y

    def calculate_pdf_y_given_x(self, pdf_x):
        # f_y_given_x = f_x_and_y / f_x and f_x_and_y = sum over u (f_x_given_u * f_y_given_u* f_u)
        pdf_u = self.s_regime.calculate_pdf_u(pdf_x)

        pdf_x_given_u = torch.transpose(self.pdf_u_given_x, 0, 1) / (pdf_u)
        pdf_x_and_y_given_u = pdf_x_given_u[:, None, :] * self.pdf_y_given_u[None, :, :]
        pdf_x_and_y = pdf_x_and_y_given_u @ pdf_u
        pdf_y_given_x = torch.transpose(pdf_x_and_y, 0, 1)
        pdf_y_given_x = pdf_y_given_x / (torch.sum(pdf_y_given_x, axis=0))
        return pdf_y_given_x

    def calculate_entropy_y_given_x(self, pdf_x, eps):
        pdf_y_given_x = self.calculate_pdf_y_given_x(pdf_x)

        entropy_y_given_x = (
            torch.sum(
                -pdf_y_given_x
                * torch.log((pdf_y_given_x + eps) / (self.config["delta_y"])),
                axis=0,
            )
            @ pdf_x
        )

        return entropy_y_given_x

    def capacity(self, pdf_x, eps=1e-20):
        entropy_y = self.calculate_entropy_y(pdf_x, eps)
        entropy_y_given_x = self.calculate_entropy_y_given_x(pdf_x, eps)
        cap = entropy_y - entropy_y_given_x
        return cap

    def capacity_like_ba(self, pdf_x):
        if self.config["sigma_1"] <= 1e-5:
            return self.f_regime.capacity_like_ba(pdf_x)
        if self.config["sigma_2"] <= 1e-5:
            return self.s_regime.capacity_like_ba(pdf_x)

        pdf_y_given_x = self.calculate_pdf_y_given_x(pdf_x)
        pdf_x_given_y = pdf_y_given_x * pdf_x
        pdf_x_given_y = torch.transpose(pdf_x_given_y, 0, 1) / (
            torch.sum(pdf_x_given_y, axis=1)
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

        # breakpoint()
        return cap

    def capacity_with_interference(self, pdf_x, pdf_x2, alphabet_x2, x2_fixed=False):
        if x2_fixed and self.pdf_y_given_x_int is not None:
            pdf_y_given_x = self.pdf_y_given_x_int
        else:
            pdf_y_given_x = self.get_pdf_y_given_x_with_interference(
                pdf_x2, alphabet_x2
            )

        entropy_y_given_x = torch.sum(
            (pdf_y_given_x * torch.log(pdf_y_given_x + 1e-20)) @ pdf_x
        )
        pdf_y = pdf_y_given_x @ pdf_x
        pdf_y = pdf_y / (torch.sum(pdf_y) + 1e-30)
        cap = entropy_y_given_x - torch.sum(pdf_y * torch.log(pdf_y + 1e-20))

        return cap

    def get_pdf_y_given_x(self):
        if self.pdf_y_given_x is None:
            pdf_y_given_x = torch.zeros(
                (self.alphabet_y.shape[0], self.alphabet_x.shape[0])
            )
            # Z1 <- Gaussian noise with variance sigma_1^2

            max_z1 = self.config["stop_sd"] * self.config["sigma_1"] ** 2
            delta_z1 = self.alphabet_x[1] - self.alphabet_x[0]
            max_z1 = max_z1 + (delta_z1 - (max_z1 % delta_z1))
            alphabet_z1 = torch.arange(-max_z1, max_z1 + delta_z1 / 2, delta_z1)

            pdf_z1 = (
                1
                / (torch.sqrt(torch.tensor([2 * torch.pi])) * self.config["sigma_1"])
                * torch.exp(-0.5 * (alphabet_z1) ** 2 / self.config["sigma_1"] ** 2)
            )
            pdf_z1 = pdf_z1 / (torch.sum(pdf_z1) + 1e-30)

            for ind, x in enumerate(self.alphabet_x):
                # U = X + Z_1
                alphabet_u = alphabet_z1 + x
                # V = phi(U)

                alphabet_v = self.nonlinear_fn(alphabet_u)
                pdf_y_given_x_and_z1 = (
                    1
                    / (
                        torch.sqrt(torch.tensor([2 * torch.pi]))
                        * self.config["sigma_2"]
                    )
                    * torch.exp(
                        -0.5
                        * (
                            (self.alphabet_y.reshape(-1, 1) - alphabet_v.reshape(1, -1))
                            ** 2
                        )
                        / self.config["sigma_2"] ** 2
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

    def get_pdf_y_given_x_with_interference(self, pdf_x2, alphabet_x2):
        pdf_y_given_x = torch.zeros(
            (self.alphabet_y.shape[0], self.alphabet_x.shape[0])
        )
        # Z1 <- Gaussian noise with variance sigma_1^2
        # X2 <- given by the user

        max_z1 = self.config["stop_sd"] * self.config["sigma_11"] ** 2
        delta_z1 = self.alphabet_x[1] - self.alphabet_x[0]
        max_z1 = max_z1 + (delta_z1 - (max_z1 % delta_z1))
        alphabet_z1 = torch.arange(-max_z1, max_z1 + delta_z1 / 2, delta_z1)

        pdf_z1 = (
            1
            / (torch.sqrt(torch.tensor([2 * torch.pi])) * self.config["sigma_11"])
            * torch.exp(-0.5 * (alphabet_z1) ** 2 / self.config["sigma_11"] ** 2)
        )
        pdf_z1 = pdf_z1 / (torch.sum(pdf_z1) + 1e-30)

        # Z_1 and X_2 are independent
        # Make sure that the pdf_x2 is normalized

        pdf_x2_and_z1 = pdf_x2[:, None] * pdf_z1[None, :]

        for ind, x in enumerate(self.alphabet_x):
            # U = X + Z_1 + X_2
            alphabet_u = alphabet_z1[None, :] + x + alphabet_x2[:, None]
            # V = phi(U)

            alphabet_v = self.nonlinear_fn(alphabet_u)
            pdf_y_given_x_z1_x2 = (
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
