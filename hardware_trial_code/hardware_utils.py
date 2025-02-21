from scipy import constants as const
import numpy as np
import matplotlib.pyplot as plt
from utils import grid_minor
import os
from nonlinearity_utils import get_nonlinear_fn_numpy


def get_noise_power(noise_figure, bandwidth):
    T = 290  # K
    N_0 = 10 * np.log10(const.k * T) + noise_figure  # dB
    P_N = N_0 + 10 * np.log10(bandwidth)  # Noise Power in dB
    P_N_linear = 10 ** (P_N / 10)  # Noise Power in linear scale (Watts)
    return P_N_linear, P_N


# G(ktanh(x/k)) = G(x - 1/3 x^3/k^2 ....), \a;pha_1 = 1, \alpha_3 = -1/3k^2
def get_k_of_tanh(iip3_power_dBm):
    iip3_power_dB = iip3_power_dBm - 30  # dBm to dB
    alpha_1 = 1
    iip3_power_lin = 10 ** (iip3_power_dB / 10)  # Watts
    # Power = V^2/(2R), R= 50 Ohms
    iip3_V = np.sqrt(2 * 50 * iip3_power_lin)

    A_0_ssa = np.sqrt(3 / 8) * iip3_V
    # IIP3 = \sqrt(4/3 |\alpha_1, \alpha_3|)
    k = iip3_V / 2
    alpha_3 = -1 / (3 * k**2)
    # This should be saturation voltage
    A_1dB = np.sqrt(0.145 * abs(alpha_1) / abs(alpha_3))

    return k, A_1dB, A_0_ssa


def get_sensitivity_analysis(noise_figure, bandwidth, iip3_power_dBm, SNR_min_dB):
    _, P_N_dB = get_noise_power(noise_figure, bandwidth)
    P_in_min_dB = P_N_dB + SNR_min_dB  # dB

    iip3_power_dB = iip3_power_dBm - 30  # dBm to dB

    P_in_max_dB = (2 * iip3_power_dB - P_N_dB) / 3  # dB

    P_in_min_linear = 10 ** (P_in_min_dB / 10)
    P_in_max_linear = 10 ** (P_in_max_dB / 10)
    V_in_min = np.sqrt(2 * 50 * P_in_min_linear)
    V_in_max = np.sqrt(2 * 50 * P_in_max_linear)

    return P_in_min_dB, P_in_max_dB, V_in_min, V_in_max


def main():
    filepath = "Hardware/"
    os.makedirs(filepath, exist_ok=True)

    noise_figure = [4.53, 3.56, 3.17, 3.18, 3.16, 2.77, 2.68]  # dB
    Gain = [15.83, 17.22, 17.35, 17.38, 17.31, 17.32, 17.32]  # dB
    power_consumption = [2.336, 5.51, 10.08, 15.92, 23.16, 31.77, 41.72]  # mWatts
    iip3_power_dBm = [-6.3, -4.92, -4.32, -2.79, -1.22, 0.07, 1.01]  # dBm
    bandwidth = [
        0.5 * 10**9,
        0.51 * 10**9,
        0.53 * 10**9,
        0.6 * 10**9,
        0.58 * 10**9,
        1 * 10**9,
        0.98 * 10**9,
    ]  # GHz

    SNR_saturation = []
    A_1db_list = []
    p_n_list = []
    for ind in range(len(noise_figure)):
        k, A_1dB, A_0_ssa = get_k_of_tanh(iip3_power_dBm[ind])
        P_N, P_N_dB = get_noise_power(noise_figure[ind], bandwidth[ind])
        p_n_list.append(P_N_dB)
        # Currently I take SNR_min as double of noise power
        SNR_min_dB = (
            10  # dB (TODO: Make sure if there is a better way to calculate this)
        )

        P_in_min_dB, P_in_max_dB, V_in_min, V_in_max = get_sensitivity_analysis(
            noise_figure[ind], bandwidth[ind], iip3_power_dBm[ind], SNR_min_dB
        )

        gain_linear_power = 10 ** (Gain[ind] / 10)
        gain_voltage = np.sqrt(2 * 50 * gain_linear_power)
        A_1dB_gain = A_1dB * gain_voltage

        config = {"nonlinearity": 4, "saturation_ssa": A_0_ssa, "smoothness_ssa": 1}

        x = np.linspace(V_in_min, 5 * k, 100)
        y = gain_voltage * k * np.tanh(x / k)

        ssa_phi = get_nonlinear_fn_numpy(config)
        ssa = gain_voltage * ssa_phi(x)
        tanh_with_A1dB = gain_voltage * A_1dB * np.tanh(x / A_1dB)

        fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)
        ax.plot(x, y, linewidth=3, label="tanh")
        ax.plot(x, tanh_with_A1dB, linewidth=3, label="tanh with A_1dB")
        ax.plot(x, ssa, linewidth=3, label="SSA")
        ax.scatter(A_1dB, A_1dB_gain, color="red", label="A_1dB+Gain")
        ax = grid_minor(ax)
        ax.legend(loc="best", fontsize=10)
        ax.set_xlabel("V_in (V)")
        ax.set_ylabel("V_out (V)")
        fig.savefig(filepath + "tanh_iip3=" + str(iip3_power_dBm[ind]) + ".png")

        # print("SNR_max: ", (P_in_max_dB - P_N_dB))
        # print("SNR_min: ", (P_in_min_dB - P_N_dB))

        P_1dB_lin = A_1dB**2 / (2 * 50)  # V^2/(2R) = P
        P_1dB_dB = 10 * np.log10(P_1dB_lin)
        SNR_saturation.append(P_1dB_dB - P_N_dB)
        A_1db_list.append(A_1dB)

    fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)
    ax.plot(power_consumption, SNR_saturation, linewidth=3)
    ax = grid_minor(ax)
    ax.set_xlabel("Power Consumption (mW)")
    ax.set_ylabel("SNR Saturation (dB)")
    fig.savefig(filepath + "SNR_saturation.png")

    fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)
    ax.plot(power_consumption, A_1db_list, linewidth=3)
    ax = grid_minor(ax)
    ax.set_xlabel("Power Consumption (mW)")
    ax.set_ylabel("A_1dB (V)")
    fig.savefig(filepath + "A_1dB.png")

    fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)
    ax.plot(power_consumption, p_n_list, linewidth=3)
    ax = grid_minor(ax)
    ax.set_xlabel("Power Consumption (mW)")
    ax.set_ylabel("Noise Power (dB)")
    fig.savefig(filepath + "Noise_power.png")


if __name__ == "__main__":
    main()
