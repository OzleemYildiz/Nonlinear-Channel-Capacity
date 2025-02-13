import matplotlib.pyplot as plt
import numpy as np
from scipy import io
import os
from utils import get_alphabet_x_y
import pandas as pd
import seaborn as sns


# Plotting the effect of sigma on the performance of the algorithm
def sigma_affect_plot():
    filepath = "Paper_Figure/Sigma_Affect/"
    name = []
    sigma_1 = [0.5, 2, 8]
    sigma_2 = [1, 1, 1]
    k = 3
    for i in range(len(sigma_1)):
        name.append(
            "_Avg_phi=3_regime=3_min_samples=64_tdm=False_lr=[0.0001]_sigma1="
            + str(sigma_1[i])
            + "_sigma2="
            + str(sigma_2[i])
            + "_gd=True_"
            + str(k)
            + "tanh"
            + str(1 / k)
            + "x"
        )

    fig1, ax1 = plt.subplots(figsize=(5, 4), tight_layout=True)
    pdf = []
    for i in range(len(name)):
        data = io.loadmat(filepath + name[i] + "/res.mat")
        power_change = data["Power_Change"]
        difference = (
            data["Capacity_without_Nonlinearity"] - data["Learned_Capacity"]
        ) / (data["Capacity_without_Nonlinearity"])
        ax1.plot(
            power_change[0],
            difference[0],
            label="$\sigma_1=$" + str(sigma_1[i]) + ", $\sigma_2=$" + str(sigma_2[i]),
            linewidth=2,
        )

        pdf.append(io.loadmat(filepath + name[i] + "/pdf.mat"))
    ax1.set_xlabel("$P_1$", fontsize=10)
    ax1.set_ylabel("$(R_L -R_O)/R_L$", fontsize=10)
    ax1.legend(loc="best", fontsize=10)
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
    ax1.set_title("$k=$" + str(k))
    filepath = filepath + "k=" + str(k) + "/"
    os.makedirs(filepath, exist_ok=True)
    fig1.savefig(filepath + "sigma_effect.png")

    plt.close()
    config = {
        "min_samples": 64,
        "regime": 3,
        "cons_type": 1,
        "sigma_1": 200,
        "sigma_2": 200,
        "nonlinearity": 3,
        "tanh_factor": 200,
        "stop_sd": 4,
    }

    # create data

    for ind, p in enumerate(power_change[0]):
        alphabet_x, _, max_x, _ = get_alphabet_x_y(config, p, k)
        alphabet_x = alphabet_x.numpy()
        alphabet_x_boundaries = alphabet_x[1:] - (alphabet_x[1] - alphabet_x[0]) / 2

        data = {}
        fig, ax2 = plt.subplots(figsize=(5, 4), tight_layout=True)
        w = 1
        for j in range(len(name)):
            try:
                pdf_x, alph_x = pdf[j]["Chng" + str(int(p * 10))]
            except:
                pdf_x, alph_x = pdf[j]["Chng" + str(int(p * 10 - 1))]

            # to make sure that comparison is fair with the same alphabet
            indices = np.digitize(alph_x, alphabet_x_boundaries)
            pdf_x_update = np.array(
                [pdf_x[indices == i].sum() for i in range(0, len(alphabet_x))]
            )
            pdf_x_update = pdf_x_update * 20000
            data = np.repeat(alphabet_x, pdf_x_update.astype(int))

            # sns.displot(data, multiple="stack", kind="kde") - this is figure plot, i dont know how to stack, maybe pandas first
            sns.histplot(
                data,
                bins=70,
                kde=True,
                ax=ax2,
                stat="probability",
                label="$\sigma_1=$"
                + str(sigma_1[j])
                + ", $\sigma_2=$"
                + str(sigma_2[j]),
            )
            # print("Updated to" + str(len(pdf_x_update)) + " from " + str(len(pdf_x)))
            # cdf = np.cumsum(pdf_x)
            # def smooth(y, box_pts):
            #     box = np.ones(box_pts) / box_pts
            #     y_smooth = np.convolve(y, box, mode="same")
            #     return y_smooth

            # pdf_x_update = smooth(pdf_x_update, 19)

            # ax.bar(
            #     alphabet_x,
            #     pdf_x_update,
            #     label="$\sigma_1=$"
            #     + str(sigma_1[j])
            #     + ", $\sigma_2=$"
            #     + str(sigma_2[j]),
            #     width=w,
            # )

            # ax.plot(
            #     alphabet_x,
            #     pdf_x_update,
            #     label="$\sigma_1=$"
            #     + str(sigma_1[j])
            #     + ", $\sigma_2=$"
            #     + str(sigma_2[j]),
            #     linewidth=2,
            # )

        ax2.set_xlabel("$x$", fontsize=10)
        ax2.set_ylabel("$p_x(x)$", fontsize=10)
        ax2.legend(loc="best", fontsize=10)
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
        ax2.set_title("$P_1=$" + str(round(p, 2)) + ", $k=$" + str(k))
        fig.savefig(filepath + "pdf_" + str(int(p * 10)) + ".png")
        plt.close()
    print("Saved in", filepath)


def plot_pdf_x():
    pass


def main():
    plt.rcParams["text.usetex"] = True
    sigma_affect_plot()


if __name__ == "__main__":
    main()
