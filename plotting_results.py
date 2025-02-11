import matplotlib.pyplot as plt
import numpy as np


# Plotting the effect of sigma on the performance of the algorithm
def sigma_affect_plot():
    filepath = "Paper_Figure/Sigma_Affect/"
    name = []
    name.append(
        "_Avg_phi=3_regime=3_min_samples=64_tdm=False_lr=[0.0001]_sigma1=1_sigma2=1_gd=True_3tanh0.3333333333333333x"
    )
    name.append(
        "_Avg_phi=3_regime=3_min_samples=64_tdm=False_lr=[0.0001]_sigma1=0.5_sigma2=1_gd=True_3tanh0.3333333333333333x"
    )
    name.append(
        "_Avg_phi=3_regime=3_min_samples=64_tdm=False_lr=[0.0001]_sigma1=1_sigma2=0.5_gd=True_3tanh0.3333333333333333x"
    )


def main():
    plt.rcParams["text.usetex"] = True

    pass


if __name__ == "__main__":
    main()
