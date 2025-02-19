import torch
import numpy as np


# It is one-one function so pdf stays the alphabet changes
def alphabet_fix_nonlinear(alphabet_x, config):
    if config["regime"] == 1:
        nonlinear_func = return_nonlinear_fn(config)
        x_change = nonlinear_func(alphabet_x)
    elif config["regime"] == 2:
        x_change = alphabet_x  # since nonlinearity is applied to the channel output
    return x_change


def return_nonlinear_fn(config, tanh_factor=None):
    if config["hardware_params_active"]:  # hardware parameters
        hn = Hardware_Nonlinear_and_Noise(config)
        nonlinear_fn = hn.nonlinear_func_torch()
    else:  # the parameters that we set
        # 0:linear
        if config["nonlinearity"] == 0:
            nonlinear_fn = lambda x: x
        #  1:nonlinear C1 (sgn(X)sqrt(abs(X)))
        elif config["nonlinearity"] == 1:
            nonlinear_fn = lambda x: torch.sign(torch.tensor(x)) * torch.sqrt(
                torch.abs(torch.tensor(x))
            )
        # 2: nonlinear C2 (sgn(X)abs(X)^{0.9})
        elif config["nonlinearity"] == 2:
            nonlinear_fn = (
                lambda x: torch.sign(torch.tensor(x))
                * torch.abs(torch.tensor(x)) ** 0.9
            )
        # 3:nonlinear tanh(X)
        elif config["nonlinearity"] == 3:
            if tanh_factor is None:
                raise ValueError("Tanh factor not defined")
            nonlinear_fn = lambda x: tanh_factor * torch.tanh(
                torch.tensor(x / tanh_factor)
            )
        # 4:nonlinear x4: x/(1 + x^4)^1/4
        elif config["nonlinearity"] == 4:  # SSA with smoothness p, saturation_ssa
            nonlinear_fn = lambda x: torch.tensor(
                x
                / (
                    (
                        1
                        + (x / config["saturation_ssa"])
                        ** (2 * config["smoothness_ssa"])
                    )
                    ** (1 / (2 * config["smoothness_ssa"]))
                )
            )
        # 5:nonlinear clipping
        elif config["nonlinearity"] == 5:
            # nonlinear_fn = lambda x: torch.ones_like(torch.tensor(x))
            nonlinear_fn = (
                lambda x: config["clipping_limit_y"]
                / config["clipping_limit_x"]
                * torch.clip(
                    torch.tensor(x),
                    -config["clipping_limit_x"],
                    config["clipping_limit_x"],
                )
            )
        else:
            raise ValueError("Nonlinearity not defined")
    return nonlinear_fn


# NOT USED
def return_derivative_of_nonlinear_fn(config, tanh_factor=None):
    # This function is numpy
    if config["hardware_params_active"]:  # hardware parameters
        hn = Hardware_Nonlinear_and_Noise(config)
        nonlinear_fn = hn.derivative_nonlinear_func_numpy()
    else:
        # 0:linear
        if config["nonlinearity"] == 0:
            nonlinear_fn = lambda x: 1
        elif config["nonlinearity"] == 1:
            nonlinear_fn = lambda x: (
                np.sign(x) / (2 * np.sqrt(np.abs(x))) if abs(x) > 1e-3 else 50
            )
        # 3:nonlinear tanh(X)
        elif config["nonlinearity"] == 3:
            if tanh_factor is None:
                raise ValueError("Tanh factor not defined")
            nonlinear_fn = lambda x: (1 / np.cosh(x / tanh_factor)) ** 2
        # 4:nonlinear x4: x/(1 + x^4)^1/4
        elif config["nonlinearity"] == 4:
            nonlinear_fn = lambda x: (
                (x / config["saturation_ssa"]) ** (2 * config["smoothness_ssa"]) + 1
            ) ** (-1 / (2 * config["smoothness_ssa"]) - 1)
            # nonlinear_fn = lambda x: 1 / (1 + x**4) ** (5 / 4)
        elif config["nonlinearity"] == 5:
            nonlinear_fn = lambda x: (
                config["clipping_limit_y"] / config["clipping_limit_x"]
                if -config["clipping_limit_x"] < x < config["clipping_limit_x"]
                else 0
            )

        else:
            raise ValueError("Derivative is not supported")

    return nonlinear_fn


def return_nonlinear_fn_numpy(config, tanh_factor=None):
    if config["hardware_params_active"]:  # hardware parameters
        hn = Hardware_Nonlinear_and_Noise(config)
        nonlinear_fn = hn.nonlinear_func_numpy()
    else:  # the parameters that we set
        # 0:linear
        if config["nonlinearity"] == 0:
            nonlinear_fn = lambda x: x
        #  1:nonlinear C1 (sgn(X)sqrt(abs(X)))
        elif config["nonlinearity"] == 1:
            nonlinear_fn = lambda x: np.sign(x) * np.sqrt(np.abs(x))
        # 2: nonlinear C2 (sgn(X)abs(X)^{0.9})
        elif config["nonlinearity"] == 2:
            nonlinear_fn = lambda x: np.sign(x) * np.abs(x) ** 0.9
        # 3:nonlinear tanh(X)
        elif config["nonlinearity"] == 3:
            if tanh_factor is None:
                raise ValueError("Tanh factor not defined")
            nonlinear_fn = lambda x: tanh_factor * np.tanh(x / tanh_factor)
        # 4:nonlinear x4: x/(1 + x^4)^1/4
        elif config["nonlinearity"] == 4:
            nonlinear_fn = lambda x: (
                x
                / (
                    (
                        1
                        + (x / config["saturation_ssa"])
                        ** (2 * config["smoothness_ssa"])
                    )
                    ** (1 / (2 * config["smoothness_ssa"]))
                )
            )

        elif config["nonlinearity"] == 5:
            nonlinear_fn = lambda x: np.clip(
                x, -config["clipping_limit_x"], config["clipping_limit_x"]
            )
        else:
            raise ValueError("Nonlinearity not defined")

    return nonlinear_fn


class Hardware_Nonlinear_and_Noise:
    def __init__(self, config):
        self.f1 = config["noise_figure1"]  # dB, noise figure for the first noise source
        self.f2 = config[
            "noise_figure2"
        ]  # dB, noise figure for the second noise source
        self.gain = config["gain"]  # dB, gain of the LNA
        self.iip3 = config["iip3"]  # dBm, input-referred third-order intercept point
        self.bandwidth = config["bandwidth"]  # Hz, bandwidth of the LNA
        self.tsamp = 1 / self.bandwidth  # s, sampling time

        """
        Calibrate the saturation level for the tanh nonlinearity
        
        The nonlinearity is given by:
            y = sqrt(gain * Esat) * tanh(vin / sqrt(Esat))
            
        where Esat is the maximum output energy per sample before the gain.
        
        The IIP3*Tsamp = 4/3 * alpha1 / alpha3 = 4 * Esat
        
        This sets Esat to dbmJ
        

        """
        self.Esat = self.iip3 + 10 * np.log10(self.tsamp / 4)

        # Convert to linear
        self.gain_lin = 10 ** (self.gain / 10)
        self.Esat_lin = 10 ** (self.Esat / 10)

        self.f1_lin = 10 ** (self.f1 / 10)
        self.f2_lin = 10 ** (self.f2 / 10)
        self.EkT_lin = 10 ** (-174 / 10)

    def nonlinear_func_numpy(self):
        return lambda x: np.sqrt(self.gain_lin * self.Esat_lin) * np.tanh(
            x / np.sqrt(self.Esat_lin)
        )

    def nonlinear_func_torch(self):
        return lambda x: torch.sqrt(
            torch.tensor(self.gain_lin * self.Esat_lin)
        ) * torch.tanh(torch.tensor(x) / torch.sqrt(torch.tensor(self.Esat_lin)))

    def derivative_nonlinear_func_numpy(self):
        return (
            lambda x: np.sqrt(self.gain_lin) / np.cosh(x / np.sqrt(self.Esat_lin)) ** 2
        )

    def get_noise_vars(self):
        self.noise1_std = np.sqrt(self.EkT_lin * self.f1_lin)
        self.noise2_std = np.sqrt(self.EkT_lin * (self.f2_lin - 1))
        return self.noise1_std, self.noise2_std

    def get_min_max_power(self, SNR_min_dB=10):
        P_N_dBm = 10 * np.log10(self.noise1_std**2) + 10 * np.log10(self.bandwidth)
        P_in_min_dBm = P_N_dBm + SNR_min_dB
        P_in_max_dBm = (2 * self.iip3 - P_N_dBm) / 3

        P_in_min_dBm = P_in_min_dBm + 10 * np.log10(self.tsamp)
        P_in_max_dBm = P_in_max_dBm + 10 * np.log10(self.tsamp)

        P_in_min_linear = 10 ** (P_in_min_dBm / 10)
        P_in_max_linear = 10 ** (P_in_max_dBm / 10)
        return P_in_min_linear, P_in_max_linear
