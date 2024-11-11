from utils import (
    read_config,
    plot_interference,
    get_interference_alphabet_x_y,
    interference_dependent_snr,
    plot_res,
    plot_R1_R2_curve,
)
import numpy as np
from gaussian_capacity import gaussian_interference_capacity
import os
from scipy import io
import matplotlib.pyplot as plt


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
        + "_delta_y="
        + str(config["delta_y"])
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

    cap_gaus_RX1 = []
    cap_gaus_RX2 = []
    lambda_power = np.linspace(0.01, 0.99, 60)
    for power in np.logspace(
        np.log10(config["min_power_cons"]),
        np.log10(config["max_power_cons"]),
        config["n_snr"],
    ):
        for lmbd in lambda_power:
            power1 = power
            power2 = lmbd * power

            print("-----------Power: ", power, "when lambda=", lmbd, "-----------")
            snr1, snr2, inr1 = interference_dependent_snr(config, power)
            alphabet_x_RX1, alphabet_y_RX1, alphabet_x_RX2, alphabet_y_RX2 = (
                get_interference_alphabet_x_y(config, power)
            )
            cap1, cap2 = gaussian_interference_capacity(
                power1,
                power2,
                config,
                alphabet_x_RX1,
                alphabet_y_RX1,
                alphabet_x_RX2,
                alphabet_y_RX2,
            )
            cap_gaus_RX1.append(cap1)
            cap_gaus_RX2.append(cap2)
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
