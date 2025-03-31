import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import yaml


# It is one-one function so pdf stays the alphabet changes
def alphabet_fix_nonlinear(alphabet_x, config):
    if config["regime"] == 1:
        nonlinear_func = get_nonlinear_fn(config)
        x_change = nonlinear_func(alphabet_x)
    elif config["regime"] == 2:
        x_change = alphabet_x  # since nonlinearity is applied to the channel output
    return x_change


def get_nonlinear_fn(config, tanh_factor=None):
    if config["dB_definition_active"]:
        non = Nonlinearity_Noise(config)
        nonlinear_fn = non.nonlinear_func_torch()
        return nonlinear_fn
    elif config["hardware_params_active"]:  # hardware parameters
        hn = Hardware_Nonlinear_and_Noise(config)
        nonlinear_fn = hn.nonlinear_func_torch()
        return nonlinear_fn
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
def get_derivative_of_nonlinear_fn(config, tanh_factor=None):
    if config["dB_definition_active"]:
        non = Nonlinearity_Noise(config)
        nonlinear_fn = non.derivative_nonlinear_func_numpy()
    # This function is numpy
    elif config["hardware_params_active"]:  # hardware parameters
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


def get_nonlinear_fn_numpy(config, tanh_factor=None):
    if config["dB_definition_active"]:
        non = Nonlinearity_Noise(config)
        nonlinear_fn = non.nonlinear_func
    elif config["hardware_params_active"]:  # hardware parameters
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


def get_inverse_nonlinearity(config, tanh_factor=None):
    if config["dB_definition_active"]:
        non = Nonlinearity_Noise(config)
        inverse_nonlinear_fn = non.get_inverse_nonlinearity()
    elif config["hardware_params_active"]:
        hn = Hardware_Nonlinear_and_Noise(config)
        inverse_nonlinear_fn = hn.get_inverse_nonlinearity()
    else:
        if config["nonlinearity"] == 3:
            inverse_nonlinear_fn = lambda x: torch.tensor(tanh_factor) * torch.atanh(
                torch.tensor(x / tanh_factor)
            )
        else:
            raise ValueError("Inverse nonlinearity is not supported")
    return inverse_nonlinear_fn


def get_derivative_of_inverse_nonlinearity(config, tanh_factor=None):
    if config["dB_definition_active"]:
        non = Nonlinearity_Noise(config)
        derivative_of_inverse_nonlinear_fn = (
            non.get_derivative_of_inverse_nonlinearity()
        )

    elif config["hardware_params_active"]:
        hn = Hardware_Nonlinear_and_Noise(config)
        derivative_of_inverse_nonlinear_fn = hn.get_derivative_of_inverse_nonlinearity()
    else:
        if config["nonlinearity"] == 3:
            derivative_of_inverse_nonlinear_fn = lambda x: (
                1 / (1 - torch.tensor(x / tanh_factor) ** 2)
            )
        else:
            raise ValueError("Derivative of inverse nonlinearity is not supported")
    return derivative_of_inverse_nonlinear_fn


def quant(distances, pdf_y_given_x, quant_locs, indices, temperature=0.01):
    # To make it differentiable, I will try with temperature
    correction_term = 10 ** np.round(np.log10(torch.min(distances)))
    weights = torch.softmax(-distances / (temperature * correction_term), dim=1)

    quant_pdf = torch.matmul(weights.T, pdf_y_given_x)

    # # Check for temperature
    # summed_pdf = torch.zeros_like(quant_locs, dtype=torch.float)
    # summed_pdf.scatter_add_(0, indices, pdf_x)
    # check_if_correct = torch.sum(abs(summed_pdf - quant_pdf)) < 10 ** (
    #     -torch.log2(torch.tensor(quant_locs.shape))
    # )

    return quant_pdf


def real_quant(quant_locs, indices, pdf):
    if len(pdf.shape) == 1:
        summed_pdf = torch.zeros((len(quant_locs)), dtype=torch.float)
        summed_pdf.scatter_add_(0, indices, pdf)
    elif len(pdf.shape) == 2:
        summed_pdf = torch.zeros((len(quant_locs), pdf.shape[1]), dtype=torch.float)
        summed_pdf.scatter_add_(0, indices.unsqueeze(1).expand(-1, pdf.shape[1]), pdf)
    elif len(pdf.shape) == 3:
        summed_pdf = torch.zeros(
            (len(quant_locs), pdf.shape[1], pdf.shape[2]), dtype=torch.float
        )
        summed_pdf.scatter_add_(
            0,
            indices.unsqueeze(1).unsqueeze(2).expand(-1, pdf.shape[1], pdf.shape[2]),
            pdf,
        )

    return summed_pdf


class Hardware_Nonlinear_and_Noise:
    def __init__(self, config):
        self.regime = config["regime"]
        self.f1 = config["noise_figure1"]  # dB, noise figure for the first noise source
        self.f2 = config[
            "noise_figure2"
        ]  # dB, noise figure for the second noise source
        self.gain = config["gain"]  # dB, gain of the LNA
        self.iip3 = config["iip3"]  # dBm, input-referred third-order intercept point
        self.bandwidth = config["bandwidth"]  # Hz, bandwidth of the LNA
        self.tsamp = 1 / self.bandwidth  # s, sampling time

        self.snr_range = config["snr_range"]
        self.SNR_min_dB = config["snr_min_dB"]
        self.gain_later = config["gain_later"]
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
        self.noise1_std, self.noise2_std = self.get_noise_vars()
        self.P_in_min_linear, self.P_in_max_linear = self.get_min_max_power()

    def nonlinear_func_numpy(self):
        if self.gain_later:
            lambda x: np.sqrt(self.Esat_lin) * np.tanh(x / np.sqrt(self.Esat_lin))
        else:
            return lambda x: np.sqrt(self.gain_lin * self.Esat_lin) * np.tanh(
                x / np.sqrt(self.Esat_lin)
            )

    def nonlinear_func_torch(self):
        if self.gain_later:
            return lambda x: torch.sqrt(torch.tensor(self.Esat_lin)) * torch.tanh(
                torch.tensor(x) / torch.sqrt(torch.tensor(self.Esat_lin))
            )
        else:
            return lambda x: torch.sqrt(
                torch.tensor(self.gain_lin * self.Esat_lin)
            ) * torch.tanh(torch.tensor(x) / torch.sqrt(torch.tensor(self.Esat_lin)))

    def derivative_nonlinear_func_numpy(self):
        if self.gain_later:
            return lambda x: 1 / np.cosh(x / np.sqrt(self.Esat_lin)) ** 2
        else:
            return (
                lambda x: np.sqrt(self.gain_lin)
                / np.cosh(x / np.sqrt(self.Esat_lin)) ** 2
            )

    def get_noise_vars(self):

        # if self.regime == 1:
        #     self.noise1_std = 0
        #     self.noise2_std = np.sqrt(self.EkT_lin * (self.f2_lin))
        # elif self.regime == 3:

        # I will not do regime 1 but
        # if I run regime 1, it will be like I ignore the noise inside the LNA so I should still calculate accordingly
        self.noise1_std = np.sqrt(self.EkT_lin * self.f1_lin)
        self.noise2_std = np.sqrt(self.EkT_lin * (self.f2_lin - 1))

        if self.gain_later:

            self.noise2_std = self.noise2_std / np.sqrt(
                self.gain_lin
            )  # since the noise is after the gain

        return self.noise1_std, self.noise2_std
    
    def get_total_noise_power(self):
        return self.noise1_std**2 + self.noise2_std**2
    
    
    def get_min_max_power(self):

        P_N_dBm = 10 * np.log10(self.noise1_std**2) + 10 * np.log10(self.bandwidth)
        P_in_min_dBm = P_N_dBm + self.SNR_min_dB
        # P_in_max_dBm = (2*self.iip3 - P_N_dBm) / 3
        P_in_max_dBm = P_in_min_dBm + self.snr_range

        P_in_min_dBm = P_in_min_dBm + 10 * np.log10(self.tsamp)
        P_in_max_dBm = P_in_max_dBm + 10 * np.log10(self.tsamp)

        P_in_min_linear = 10 ** (P_in_min_dBm / 10)
        P_in_max_linear = 10 ** (P_in_max_dBm / 10)

        # elif self.regime == 1:
        #     P_in_min_dBm = self.SNR_min_dB + 10 * np.log10(self.EkT_lin)
        #     P_in_max_dBm = P_in_min_dBm + self.snr_range

        #     P_in_min_linear = 10 ** (P_in_min_dBm / 10)
        #     P_in_max_linear = 10 ** (P_in_max_dBm / 10)

        # This is not a must but to keep the mappings similar for a comparison
        # if self.gain_later:
        #     P_in_min_linear = P_in_min_linear / self.gain_lin
        #     P_in_max_linear = P_in_max_linear / self.gain_lin

        return P_in_min_linear, P_in_max_linear

    def get_power_fixed_from_SNR(self, fixed_power):
        # if self.regime == 3:
        P_N_dBm = 10 * np.log10(self.noise1_std**2) + 10 * np.log10(self.bandwidth)
        P_in_dBm = fixed_power + P_N_dBm
        P_in_dBm = P_in_dBm + 10 * np.log10(self.tsamp)
        # elif self.regime == 1:
        #     P_in_dBm = fixed_power + 10 * np.log10(self.EkT_lin)
        return 10 ** (P_in_dBm / 10)

    def get_inverse_nonlinearity(self):
        # return (
        #     lambda x: 1
        #     / torch.sqrt(torch.tensor(self.gain_lin * self.Esat_lin))
        #     * torch.tanh(torch.tensor(x) / torch.sqrt(torch.tensor(self.Esat_lin)))
        # )
        if self.gain_later:
            return torch.sqrt(torch.tensor(self.Esat_lin)) * torch.atanh(
                torch.tensor(x) / torch.sqrt(torch.tensor(self.Esat_lin))
            )
        else:
            return lambda x: torch.sqrt(torch.tensor(self.Esat_lin)) * torch.atanh(
                torch.tensor(x)
                / torch.sqrt(torch.tensor(self.gain_lin * self.Esat_lin))
            )

    def get_derivative_of_inverse_nonlinearity(self):
        # return lambda x: -1 / (
        #     torch.sqrt(torch.tensor(self.gain_lin))
        #     * self.Esat_lin
        #     * torch.sinh(torch.tensor(x) / torch.sqrt(torch.tensor(self.Esat_lin))) ** 2
        # )
        # return (
        #     lambda x: torch.sqrt(torch.tensor(self.gain_lin))
        #     * self.Esat_lin
        #     / (self.gain_lin * self.Esat_lin - x**2)
        # )
        if self.gain_later:
            return lambda x: 1 / (
                1 - torch.tensor(x) ** 2 / torch.tensor(self.Esat_lin)
            )
        else:
            return lambda x: 1 / (
                torch.sqrt(torch.tensor(self.gain_lin))
                * (
                    1
                    - torch.tensor(x) ** 2
                    / (torch.tensor(self.gain_lin * self.Esat_lin))
                )
            )
        # INR: interference to noise_1 ratio
    def update_config(self, config, pp=False):
        if pp:
            config["sigma_1"] = self.noise1_std
            config["sigma_2"] = self.noise2_std

        config["sigma_11"] = self.noise1_std
        config["sigma_22"] = self.noise2_std
        config["sigma_12"] = self.noise2_std
        config["sigma_21"] = self.noise1_std
        config["E_sat"] = self.Esat_lin
        return config

class Nonlinearity_Noise:
    def __init__(self, config):
        # No gain here, it's adjusted by N_2

        self.N_1 = config["N_1"]  # dB
        self.N_2 = config["N_2"]
        self.saturation_to_noise = config["Saturation_to_Noise"]
        self.saturation = self.N_1 + self.saturation_to_noise
        
        self.sigma_1 = np.sqrt(10 ** (self.N_1 / 10))
        self.sigma_2 = np.sqrt(10 ** (self.N_2 / 10))
        
        self.Esat_lin = 10 ** (self.saturation / 10) 

    def update_config(self, config, pp=False):
        if pp:
            config["sigma_1"] = self.sigma_1
            config["sigma_2"] = self.sigma_2

        config["sigma_11"] = self.sigma_1
        config["sigma_22"] = self.sigma_2
        config["sigma_12"] = self.sigma_2
        config["sigma_21"] = self.sigma_1

        config["multiplying_factor"] = 1
        return config

    def get_power_fixed_from_SNR(self, dB):
        snr = self.N_1 + dB
        return 10 ** (snr / 10) 


    def nonlinear_func_numpy(self):

        return lambda x: np.sqrt(self.Esat_lin) * np.tanh(x / np.sqrt(self.Esat_lin))

    def nonlinear_func_torch(self):

        return lambda x: torch.sqrt(torch.tensor(self.Esat_lin)) * torch.tanh(
            torch.tensor(x) / torch.sqrt(torch.tensor(self.Esat_lin))
        )

    def derivative_nonlinear_func_numpy(self):
        return lambda x: 1 / np.cosh(x / np.sqrt(self.Esat_lin)) ** 2

    def get_inverse_nonlinearity(self):
        return torch.sqrt(torch.tensor(self.Esat_lin)) * torch.atanh(
            torch.tensor(x) / torch.sqrt(torch.tensor(self.Esat_lin))
        )

    def get_derivative_of_inverse_nonlinearity(self):
        return lambda x: 1 / (1 - torch.tensor(x) ** 2 / torch.tensor(self.Esat_lin))

    def get_total_noise_power(self):
        return self.sigma_1**2 + self.sigma_2**2


def read_conf(args_name):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=args_name,
        help="Configure of post processing",
    )
    args = parser.parse_args()
    config = yaml.load(open("args/" + args.config, "r"), Loader=yaml.Loader)
    return config


def test_nonlinearity():

    sat_points = []  # lin
    nf1s = []  # lin
    iip3 = []
    for i in np.linspace(1, 7, 7):
        str_c = "arguments" + str(int(i)) + ".yml"
        config = read_conf(str_c)
        hn = Hardware_Nonlinear_and_Noise(config)
        sat_points.append(hn.Esat)
        noise1_std, noise2_std = hn.get_noise_vars()

        nf1s.append(10 * np.log10(noise1_std**2))
        iip3.append(hn.iip3)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), tight_layout=True)
    ax1.scatter(iip3, sat_points)
    ax1.set_xlabel("IIP3 (dBm)")
    ax1.set_ylabel("Saturation Point Power (dBm)")
    ax1.grid(
        visible=True,
        which="major",
        axis="both",
        color="lightgray",
        linestyle="-",
        linewidth=0.5,
    )
    plt.minorticks_on()
    ax1.grid(
        visible=True,
        which="minor",
        axis="both",
        color="gainsboro",
        linestyle=":",
        linewidth=0.5,
    )
    ax2.scatter(iip3, nf1s)
    ax2.set_xlabel("IIP3 (dBm)")
    ax2.set_ylabel("Noise 1 Power (dBm)")
    ax2.grid(
        visible=True,
        which="major",
        axis="both",
        color="lightgray",
        linestyle="-",
        linewidth=0.5,
    )
    plt.minorticks_on()
    ax2.grid(
        visible=True,
        which="minor",
        axis="both",
        color="gainsboro",
        linestyle=":",
        linewidth=0.5,
    )

    os.makedirs("Hardware/", exist_ok=True)
    fig.savefig("Hardware/sat_nf1_dB.png")
    plt.close()


# def test_parameter_switch():
# for i in np.linspace(1, 7, 7):
#     str_c = "arguments" + str(int(i)) + ".yml"
#     config = read_conf(str_c)
#     hn = Hardware_Nonlinear_and_Noise(config)
#     print("IIP3: ", hn.iip3)
#     print("Saturation to Noise: ", hn.saturation_to_noise_1_dBm)
#     print("Min Signal to Noise: ", hn.min_signal_to_noise_1_dB)
#     print("Max Signal to Noise: ", hn.max_signal_to_noise_1_dB)
#     print("Noise 1: ", hn.noise_1_dB)
#     print("Noise 2: ", hn.noise_2_dB)

#     print("--------------------------------------")


if __name__ == "__main__":
    test_nonlinearity()
    # test_parameter_switch()
