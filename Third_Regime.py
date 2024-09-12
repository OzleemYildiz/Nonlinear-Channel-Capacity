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

        self.s_regime = Second_Regime(alphabet_x, config, power)
        self.alphabet_v = self.nonlinear_fn(self.s_regime.alphabet_u)
        self.pdf_u_given_x = self.s_regime.calculate_pdf_u_given_x()

        self.f_regime = First_Regime(alphabet_x, alphabet_y, config, power)
        self.f_regime.set_alphabet_v(
            self.alphabet_v
        )  # since Z_1 is not 0, we define new U and V
        self.pdf_y_given_v = self.f_regime.calculate_pdf_y_given_v()
        self.pdf_y_given_u = self.pdf_y_given_v

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
        pdf_x_given_u = torch.transpose(self.pdf_u_given_x, 0, 1) / pdf_u
        pdf_x_and_y_given_u = pdf_x_given_u[:, None, :] * self.pdf_y_given_u[None, :, :]
        pdf_x_and_y = pdf_x_and_y_given_u @ pdf_u
        pdf_y_given_x = torch.transpose(pdf_x_and_y, 0, 1)
        pdf_y_given_x = pdf_y_given_x / torch.sum(pdf_y_given_x, axis=0)
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
        pdf_y_given_x = self.calculate_pdf_y_given_x(pdf_x)
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
