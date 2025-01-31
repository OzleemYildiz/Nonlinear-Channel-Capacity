from utils import (
    read_config,
    plot_interference,
    get_interference_alphabet_x_y,
    interference_dependent_snr,
    plot_R1_R2_curve,
)
import numpy as np
from gaussian_capacity import gaussian_interference_capacity
import os
from scipy import io
import matplotlib.pyplot as plt

from gaussian_capacity import gaus_interference_R1_R2_curve


def define_save_location(config):
    save_location = (
        config["output_dir"]
        + "-Gaus"
        + "/"
        + config["cons_str"]
        + "_phi="
        + str(config["nonlinearity"])
        + "_regime="
        + str(config["regime"])
        # + "_int_ratio="
        # + str(config["int_ratio"])
        + "_min_pow_1="
        + str(config["min_power_cons"])
        + "_power_2="
        + str(config["power_2"])
        + "_int_ratio="
        + str(config["int_ratio"])
        + "_max_iter="
        + str(config["max_iter"])
        + "_lr="
        + str(config["lr"])
    )
    if config["regime"] == 1:
        save_location = (
            save_location
            + "_sigma22="
            + str(config["sigma_22"])
            + "_sigma12="
            + str(config["sigma_12"])
        )

    save_location = save_location + "/"
    return save_location


def main():
    # TODO: First regime implementation- Then think of the second and third regime
    # ----System Model--- Z channel
    # Y1 = Phi(X1 + AX2 + N11)+ N12
    # Y2 = Phi(X2  + N21)+ N22

    config = read_config(args_name="arguments-interference1.yml")
    print(
        "**** AWGN Interference Channel with Nonlinearity: ",
        config["nonlinearity"],
        "for Regime: ",
        config["regime"],
        "****",
    )
    save_location = define_save_location(config)
    os.makedirs(save_location, exist_ok=True)
    print("Save Location: ", save_location)

    power = config["min_power_cons"]  #

    cap_gaus_RX1, cap_gaus_RX2 = gaus_interference_R1_R2_curve(config, power)  #
    io.savemat(save_location + "cap_gaus_RX1.mat", {"cap_gaus_RX1": cap_gaus_RX1})
    io.savemat(save_location + "cap_gaus_RX2.mat", {"cap_gaus_RX2": cap_gaus_RX2})
    plt.figure()
    plt.plot(cap_gaus_RX1, cap_gaus_RX2)
    plt.xlabel("Rate of RX1")
    plt.ylabel("Rate of RX2")
    plt.grid()
    plt.savefig(save_location + "rate_curve.png")
    plt.close()


if __name__ == "__main__":
    main()
