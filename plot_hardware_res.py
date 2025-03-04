import matplotlib.pyplot as plt
from scipy import io
from utils import grid_minor
import os
import seaborn as sns
from nonlinearity_utils import real_quant
import torch
import numpy as np
import pandas as pd


def plot_different_bits():
    # folder = "Paper_Figure/Hardware-Params/ADC-PP/"
    # main_run = "_Avg_regime=1_min_samples=2000_tdm=False_lr=[1e-05]_gd=True_hardware_nf1=2.68_nf2=15_bw=980000000_iip3=1.01_gain=17.32_ADC_bits="
    # bound = "Res-PP-H/_Avg_regime=1_min_samples=1000_tdm=False_lr=[1e-05]_hardware_nf1=2.68_nf2=15_bw=980000000_iip3=1.01_gain=17.32_ADC_bits="
    folder = "Paper_Figure/Regime 3 Hardware/PP/"
    main_run = "_Avg_regime=3_min_samples=500_tdm=False_lr=[1e-05]_gd=True_hardware_nf1=4.53_nf2=15_bw=500000000_iip3=-6.3_gain=15.83_ADC_bits="
    bound = "Res-PP-B/_Avg_regime=3_min_samples=500_tdm=False_lr=[1e-05]_hardware_nf1=4.53_nf2=15_bw=500000000_iip3=-6.3_gain=15.83_ADC_bits="

    fig, ax = plt.subplots()
    color = ["b", "g", "r", "c", "m", "y", "k"]
    for i in range(3, 7):
        if i == 3:
            main_run_2 = "_Avg_regime=3_min_samples=100_tdm=False_lr=[1e-05]_gd=True_hardware_nf1=4.53_nf2=15_bw=500000000_iip3=-6.3_gain=15.83_ADC_bits="
        else:
            main_run_2 = main_run
        name = folder + main_run + str(i) + "/res.mat"
        data = io.loadmat(name)
        power_change = data["Power_Change"]
        bounds = io.loadmat(bound + str(i) + "/res.mat")

        if i == 3:
            ax.plot(
                power_change[0],
                data["Capacity_without_Nonlinearity"][0],
                label="Linear",
                linewidth=3,
                linestyle="dotted",
                color="k",
            )
        ax.plot(
            power_change[0],
            data["Learned_Capacity"][0],
            label="Learned b =" + str(i),
            linewidth=3,
            linestyle="solid",
            color=color[i - 3],
        )
        ax.plot(
            power_change[0],
            data["Gaussian_Capacity"][0],
            label="Gaussian b =" + str(i),
            linewidth=3,
            color=color[i - 3],
            linestyle="dashed",
        )
        ax.plot(
            bounds["Power_Change"][0],
            bounds["MMSE_Minimum"][0],
            label="MMSE b =" + str(i),
            linewidth=3,
            linestyle="dashdot",
            marker="v",
            color=color[i - 3],
        )
    ax.set_xlabel("Power Change")
    ax.set_ylabel("Capacity")
    ax.legend(loc="best")
    ax = grid_minor(ax)
    ax.set_title(
        "Regime 1: IIP3 =-6.3, Gain = 15.83, BW = 0.5 GHz, NF1 = 4.53 , NF2 = 15"
    )
    os.makedirs(folder + "/PP/Plots/", exist_ok=True)
    fig.savefig(folder + "/PP/Plots/" + "ADC_bits.png")
    print("ADC bits plot saved in ", folder + "Plots/")
    plt.close()


def plot_power_consumption():
    # folder = "Paper_Figure/Hardware-Params/ADC-PP/"
    folder = "Paper_Figure/Regime 3 Hardware/"
    noise_figure = [
        4.53,
        # 3.56,
        3.17,
        3.18,
        3.16,
        2.77,
        2.68,
    ]  # dB
    Gain = [
        15.83,
        # 17.22,
        17.35,
        17.38,
        17.31,
        17.32,
        17.32,
    ]  # dB
    power_consumption = [
        2.336,
        # 5.51,
        10.08,
        15.92,
        23.16,
        31.77,
        41.72,
    ]  # mWatts
    noise_figure2 = 15
    iip3_power_dBm = [
        -6.3,
        # -4.92,
        -4.32,
        -2.79,
        -1.22,
        0.07,
        1.01,
    ]  # dBm - Real one is this

    bandwidth = [
        0.5 * 10**9,
        #  0.51 * 10**9,
        0.53 * 10**9,
        0.6 * 10**9,
        0.58 * 10**9,
        1 * 10**9,
        0.98 * 10**9,
    ]  # GHz
    bits = 5
    hold_max = []
    for i in range(len(noise_figure)):
        main_run = (
            # "_Avg_regime=1_min_samples=2000_tdm=False_lr=[1e-05]_gd=True_hardware_nf1="
            "_Avg_regime=3_min_samples=100_tdm=False_lr=[1e-05]_gd=True_hardware_nf1="
            + str(noise_figure[i])
            + "_nf2="
            + str(noise_figure2)
            + "_bw="
            + str(int(bandwidth[i]))
            + "_iip3="
            + str(iip3_power_dBm[i])
            + "_gain="
            + str(Gain[i])
            + "_ADC_bits="
            + str(bits)
        )
        name = folder + main_run + "/res.mat"
        data = io.loadmat(name)
        hold_max.append(max(data["Learned_Capacity"][0]))
    fig, ax = plt.subplots()
    ax.plot(power_consumption, hold_max)
    ax.set_xlabel("Power Consumption (mW)")
    ax.set_ylabel(" Max Capacity")
    ax = grid_minor(ax)
    ax.set_title("Regime 1, ADC bits = " + str(bits))
    os.makedirs(folder + "/PP/Plots/", exist_ok=True)
    fig.savefig(folder + "/PP/Plots/" + "Power_consumption_bits=" + str(bits) + ".png")
    print("Power consumption plot saved in ", folder + "Plots/")
    plt.close()


def plot_x_y_marginal():
    chosen = "Paper_Figure/Hardware-Params/ADC-PP/_Avg_regime=1_min_samples=2000_tdm=False_lr=[1e-05]_gd=True_hardware_nf1=2.68_nf2=15_bw=980000000_iip3=1.01_gain=17.32_ADC_bits=3"

    pdfs = io.loadmat(chosen + "/pdf_y_GD_power=99.99999999999997.mat")

    pdf_x = pdfs["pdf_x"].reshape(-1)
    alph_x = pdfs["alph_x"].reshape(-1)
    q_pdf_y = pdfs["q_pdf_y"]
    q_alph_y = pdfs["q_alph_y"].reshape(-1)
    pdf_y = pdfs["pdf_y"]
    alph_y = pdfs["alph_y"].reshape(-1)
    pdf_y_given_x = pdfs["pdf_y_given_x"]

    indices = np.argmin(abs(alph_y[:, None] - q_alph_y[None, :]), axis=1)
    q_pdf_y_given_x = real_quant(
        torch.tensor(q_alph_y), torch.tensor(indices), torch.tensor(pdf_y_given_x)
    ).numpy()
    # q_pdf_y_given_x = real_quant()

    sns.set_theme(style="ticks")

    n_samp = 10 ** (-round(np.log10(np.min(pdf_x))))
    pdf_x_update = (pdf_x * n_samp).astype(int)
    xxx = np.repeat(alph_x, pdf_x_update)
    y_sample = []
    for ind, x in enumerate(alph_x):
        y_samp_rep = np.round(q_pdf_y_given_x[:, ind] * pdf_x_update[ind]).astype(int)

        if np.sum(y_samp_rep) != pdf_x_update[ind]:
            # numerical thingy, I just added to the biggest one
            st1 = np.where(y_samp_rep != 0)[0]
            choose_ind = np.argmax(y_samp_rep[st1])
            if abs(pdf_x_update[ind] - np.sum(y_samp_rep)) > 1:
                breakpoint()
            y_samp_rep[st1[choose_ind]] += pdf_x_update[ind] - np.sum(y_samp_rep)

        y_sample.append(np.repeat(q_alph_y, y_samp_rep))

    df = pd.DataFrame({"Input": xxx, "Output": np.concatenate(y_sample)})
    # g = sns.jointplot(data=df, x="x", y="y", kind="hist", marginal_kws=dict(bins=100))
    # plt.show()

    g = sns.JointGrid(data=df, x="Input", y="Output", marginal_ticks=True, height=10)
    cax = g.figure.add_axes([0.15, 0.55, 0.02, 0.2])
    g.plot_joint(
        sns.histplot,
        cmap="mako",
        pmax=2.0,
        cbar=True,
        cbar_ax=cax,
        stat="probability",
        alpha=1,
        bins=30,
        # discrete=True,
    )
    g.plot_marginals(
        sns.histplot,
        color="#03012d",
        stat="probability",
        element="bars",
        bins=30,
        common_norm=True,  # Ensures total probability sums to 1
        # discrete=True,
    )
    # Add **grid lines** (including minor ones) for clarity
    g.ax_joint.grid(True, which="both", linestyle="--", alpha=0.5)  # Joint plot grid
    g.ax_marg_x.grid(True, which="both", linestyle=":", alpha=0.5)  # Top marginal grid
    g.ax_marg_y.grid(
        True, which="both", linestyle=":", alpha=0.5
    )  # Right marginal grid
    g.ax_joint.set_title("Joint Probability Histogram", fontsize=16)
    g.ax_joint.set_xlabel("X Values", fontsize=14)
    g.ax_joint.set_ylabel("Y Values", fontsize=14)
    g.ax_joint.tick_params(axis="both", labelsize=12)  # Joint plot ticks
    g.ax_marg_x.tick_params(axis="x", labelsize=10)  # Marginal X ticks
    g.ax_marg_y.tick_params(axis="y", labelsize=10)  # Marginal Y ticks

    g.figure.savefig(
        "Paper_Figure/Hardware-Params/ADC-PP/Plots/joint_histogram.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main():
    plt.rcParams["text.usetex"] = True
    plot_different_bits()
    plot_power_consumption()
    # plot_x_y_marginal()


if __name__ == "__main__":
    main()
