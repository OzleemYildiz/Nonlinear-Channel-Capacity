from utils_interference import (
    change_parameters_range,
    get_run_parameters,
    get_updated_params_for_hd,
    get_int_regime,
    fix_config_for_hd,
)
from gaussian_capacity import gaussian_interference_capacity
from utils import read_config, grid_minor
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import io


def main(read_from_file, args_number, pw1_list):
    config = read_config("arguments-interference" + str(args_number) + ".yml")
    filename = (
        "Gaussian_Capacity/"
        + "_hw_gain="
        + str(config["gain"])
        + "_iip3="
        + str(config["iip3"])
        + "_bw="
        + str(config["bandwidth"])
        + "/"
    )
    os.makedirs(filename, exist_ok=True)
    change_range, config = change_parameters_range(config)
    ratio_pw = np.linspace(0.01, 0.99, 20)
    cap_gaus_RX1 = np.zeros((len(change_range), len(ratio_pw)))
    cap_gaus_RX2 = np.zeros((len(change_range), len(ratio_pw)))

    for ind_c, chng in enumerate(change_range):  # It has to be PW1
        power1, power2, int_ratio, tanh_factor, tanh_factor2, res_str, res_str_run = (
            get_run_parameters(config, chng)
        )
        if config["hardware_params_active"]:
            power1, power2, config = get_updated_params_for_hd(config, power1, power2)
        else:
            config["multiplying_factor"] = 1

        for ind_r, ratio in enumerate(ratio_pw):
            power2_upd = ratio * power2
            reg_RX1, reg_RX2 = get_int_regime(
                config, power1, power2_upd, int_ratio, tanh_factor, tanh_factor2
            )

            cap1, cap2 = gaussian_interference_capacity(
                reg_RX1,
                reg_RX2,
                int_ratio,
                tin_active=True,
            )
            cap_gaus_RX1[ind_c, ind_r] = cap1.numpy().item()
            cap_gaus_RX2[ind_c, ind_r] = cap2.numpy().item()

        if config["hardware_params_active"]:
            config = fix_config_for_hd(config)
            power2 = power2 / 10 ** (config["multiplying_factor"])

        res = io.loadmat(read_from_file + "/pw1=" + str(pw1_list[ind_c]) + "/res.mat")
        fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)
        ax.plot(
            cap_gaus_RX1[ind_c, :],
            cap_gaus_RX2[ind_c, :],
            label="Gaussian",
            linewidth=3,
            color="b",
        )
        ax.plot(
            res["R1"]["Learned"][0][0][0],
            res["R2"]["Learned"][0][0][0],
            label="Learned",
            linewidth=3,
            color="r",
        )
        ax.set_xlabel("Rate User 1")
        ax.set_ylabel("Rate User 2")
        ax.legend(loc="best")
        ax = grid_minor(ax)
        ax.set_title(
            "Regime 3: IIP3 = "
            + "{:.1f}".format(config["iip3"])
            + ", Gain = "
            + str(config["gain"])
            + ", BW = "
            + str(config["bandwidth"] / 1e6)
            + " MHz, PW1 = "
            + "{:.1e}".format(pw1_list[ind_c])
            + ", PW2 = "
            + "{:.1e}".format(power2)
        )
        fig.savefig(
            filename
            + "Regime_3_hw_gain="
            + str(config["gain"])
            + "_iip3="
            + str(config["iip3"])
            + "_bw="
            + str(config["bandwidth"] / 1e6)
            + "_pw1="
            + str(pw1_list[ind_c])
            + ".png",
            bbox_inches="tight",
        )
        print("Plot saved in ", filename)
        plt.close()


if __name__ == "__main__":
    read_from_file = "Paper_Figure/Regime 3 Hardware/Int/New/BothOptimize-NoADC/Avg_hw_gain=17.32_iip3=1.01_bw=980000000_NF1=2.68_NF2=15_regime=3_N=100_max_iter=10000_lr=[0.0001]_pw1_fixed_SNR=5_pw2=2.333458062281014e-17_a=1_k=1.794142634330711e-05_k2=1.794142634330711e-05"
    pw_list = [2.3334580622810138e-18, 7.379042301291046e-17, 2.3334580622810045e-15]
    args_number = 1
    plt.rcParams["text.usetex"] = True

    main(read_from_file, args_number, pw_list)
