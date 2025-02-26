import matplotlib.pyplot as plt
from scipy import io
from utils import grid_minor
import os


def plot_different_bits():
    folder = "Paper_Figure/Hardware-Params/ADC-PP/"
    main_run = "_Avg_regime=1_min_samples=2000_tdm=False_lr=[1e-05]_gd=True_hardware_nf1=2.68_nf2=15_bw=980000000_iip3=1.01_gain=17.32_ADC_bits="

    fig, ax = plt.subplots()

    for i in range(3, 9):
        name = folder + main_run + str(i) + "/res.mat"
        data = io.loadmat(name)
        power_change = data["Power_Change"]

        if i == 3:
            ax.plot(
                power_change[0],
                data["Capacity_without_Nonlinearity"][0],
                label="Linear",
                linewidth=3,
                linestyle=":",
            )
        ax.plot(
            power_change[0],
            data["Learned_Capacity"][0],
            label="b =" + str(i),
            linewidth=3,
        )
    ax.set_xlabel("Power Change")
    ax.set_ylabel("Capacity")
    ax.legend(loc="best")
    ax = grid_minor(ax)
    ax.set_title("Regime 1: IIP3 = 1.01, Gain = 17.32, BW = 0.98 GHz, NF2 = 15")
    os.makedirs(folder + "Plots/", exist_ok=True)
    fig.savefig(folder + "Plots/" + "ADC_bits.png")
    print("ADC bits plot saved in ", folder + "Plots/")


def plot_power_consumption():
    folder = "Paper_Figure/Hardware-Params/ADC-PP/"
    noise_figure = [4.53, 3.56, 3.17, 3.18, 3.16, 2.77, 2.68]  # dB
    Gain = [15.83, 17.22, 17.35, 17.38, 17.31, 17.32, 17.32]  # dB
    power_consumption = [2.336, 5.51, 10.08, 15.92, 23.16, 31.77, 41.72]  # mWatts
    noise_figure2 = 15
    # iip3_power_dBm = [-6.3, -4.92, -4.32, -2.79, -1.22, 0.07, 1.01]  # dBm - Real one is this
    iip3_power_dBm = [-4.32, -4.92, -4.32, -2.79, -1.22, 0.07, 1.01]  # dBm
    bandwidth = [
        0.5 * 10**9,
        0.51 * 10**9,
        0.53 * 10**9,
        0.6 * 10**9,
        0.58 * 10**9,
        1 * 10**9,
        0.98 * 10**9,
    ]  # GHz
    bits = 3
    hold_max = []
    for i in range(len(noise_figure)):
        main_run = (
            "_Avg_regime=1_min_samples=2000_tdm=False_lr=[1e-05]_gd=True_hardware_nf1="
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
    ax.set_title("Regime 1, ADC bits = 3")
    os.makedirs(folder + "Plots/", exist_ok=True)
    fig.savefig(folder + "Plots/" + "Power_consumption.png")
    print("Power consumption plot saved in ", folder + "Plots/")


def main():
    plt.rcParams["text.usetex"] = True
    # plot_different_bits()
    plot_power_consumption()


if __name__ == "__main__":
    main()
