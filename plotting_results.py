import matplotlib.pyplot as plt
import numpy as np
from scipy import io


# Plotting the effect of sigma on the performance of the algorithm
def sigma_affect_plot():
    filepath = "Paper_Figure/Sigma_Affect/"
    name = []
    sigma_1 = [1, 0.5, 1, 0.5]
    sigma_2 = [1, 1, 0.5, 0.5]
    name.append(
        "_Avg_phi=3_regime=3_min_samples=64_tdm=False_lr=[0.0001]_sigma1="
        + sigma_1[0]
        + "_sigma2="
        + sigma_2[0]
        + "_gd=True_3tanh0.3333333333333333x"
    )
    name.append(
        "_Avg_phi=3_regime=3_min_samples=64_tdm=False_lr=[0.0001]_sigma1="
        + sigma_1[1]
        + "_sigma2="
        + sigma_2[1]
        + "_gd=True_3tanh0.3333333333333333x"
    )
    name.append(
        "_Avg_phi=3_regime=3_min_samples=64_tdm=False_lr=[0.0001]_sigma1="
        + sigma_1[2]
        + "_sigma2="
        + sigma_2[2]
        + "_gd=True_3tanh0.3333333333333333x"
    )
    name.append(
        "_Avg_phi=3_regime=3_min_samples=64_tdm=False_lr=[0.0001]_sigma1="
        + sigma_1[3]
        + "_sigma2="
        + sigma_2[3]
        + "_gd=True_3tanh0.3333333333333333x"
    )

    fig1, ax1 = plt.subplots(figsize=(5, 4), tight_layout=True)
    fig2, ax2 = plt.subplots(figsize=(5, 4), tight_layout=True)

    for i in range(len(name)):
        data = io.loadmat(filepath + name[i] + "/res.mat")
        power_change = data["power_change"]
        breakpoint()


def main():
    plt.rcParams["text.usetex"] = True
    sigma_affect_plot()


if __name__ == "__main__":
    main()
