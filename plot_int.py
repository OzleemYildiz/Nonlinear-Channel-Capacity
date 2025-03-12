import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import io
from utils import grid_minor


def plot_R1_R2_curve_for_different_power(main_folder, sub_folder, iip3):
    R1 = {}
    R2 = {}
    for dirpath, dirnames, _ in os.walk(sub_folder):
        for dirname in dirnames:
            res = io.loadmat(sub_folder + "/" + dirname + "/res.mat")
            key = "{:.1e}".format(float(dirname[4:]))
            R1[key] = res["R1"]["Learned"][0][0][0]
            R2[key] = res["R2"]["Learned"][0][0][0]

    fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)
    for keys in R1.keys():
        plt.plot(R1[keys], R2[keys], label="$P_1=$" + str(keys))
    ax.legend(loc="best")
    ax.set_xlabel("Rate User 1")
    ax.set_ylabel("Rate User 2")
    ax.set_title("Regime 3 with IIP3: ")
    ax = grid_minor(ax)
    os.makedirs(main_folder + "Plots", exist_ok=True)
    fig.savefig(main_folder + "Plots/R1_R2_curve_iip3=" + str(iip3) + ".png")
    plt.close()
    print("Plot saved in ", main_folder + "Plots")


def plot_low_power_rate_curve(main_folder, range_power):

    fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)
    for dirpath, dirnames, _ in os.walk(main_folder):
        for dirname in dirnames:
            if dirname != "Plots":
                for sub_dirpath, sub_dirnames, _ in os.walk(main_folder + dirname):
                    for sub_dirname in sub_dirnames:

                        if float(sub_dirname[4:]) / range_power < 10:
                            res = io.loadmat(
                                main_folder + dirname + "/" + sub_dirname + "/res.mat"
                            )

                            ind_s = dirname.find("iip3") + 5
                            ind_e = dirname.find("_bw")

                            key = dirname[ind_s:ind_e]
                            plt.plot(
                                res["R1"]["Learned"][0][0][0],
                                res["R2"]["Learned"][0][0][0],
                                label="IIP3 = " + key,
                            )
    ax.legend(loc="best")
    ax.set_xlabel("Rate User 1")
    ax.set_ylabel("Rate User 2")
    ax.set_title("Regime 3 with Different Hardware Params ")
    ax = grid_minor(ax)
    os.makedirs(main_folder + "Plots", exist_ok=True)
    fig.savefig(main_folder + "Plots/R1_R2_curve_comp.png")
    plt.close()
    print("Plot saved in ", main_folder + "Plots")


def main():
    plt.rcParams.update({"font.size": 14})
    plt.rcParams["text.usetex"] = True
    main_folder = "Paper_Figure/Regime 3 Hardware/Int/New/BothOptimize-NoADC/"
    sub_folder = (
        main_folder
        + "Avg_hw_gain=15.83_iip3=-6.3_bw=500000000_NF1=4.53_NF2=15_regime=3_N=100_max_iter=10000_lr=[0.0001]_pw1_fixed_SNR=5_pw2=3.5727283815192956e-17_a=1_k=1.082642326745063e-05_k2=1.082642326745063e-05"
    )
    iip3 = -6.3

    # plot_R1_R2_curve_for_different_power(main_folder, sub_folder, iip3)
    range_power = 1e-18
    plot_low_power_rate_curve(main_folder, range_power)


if __name__ == "__main__":
    main()
