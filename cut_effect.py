from bounds import (
    sdnr_new_with_erf,
    upper_bound_peak,
    upper_bound_peak_power,
    sdnr_new_with_erf_nopowchange,
)
from utils import read_config
import numpy as np
import matplotlib.pyplot as plt
import os


def main():
    config = read_config()
    x_cut_list = np.linspace(2, 300, 30)

    peak_max_power = []
    peak_cap = []
    sdnr_cap = []
    sdnr_power = []
    for x_c in x_cut_list:
        print(f"----------------x_cut: {x_c}----------------")
        config["clipping_limit_x"] = x_c
        config["clipping_limit_y"] = x_c

        peak_max_power.append(upper_bound_peak_power(x_c))
        peak_cap.append(upper_bound_peak(peak_max_power[-1] + 10, config))
        sdnr_cap.append(sdnr_new_with_erf(peak_max_power[-1], config))
        found = False
        power = peak_max_power[-1]
        old_cap = sdnr_cap[-1]
        while not found:
            new_cap = sdnr_new_with_erf_nopowchange(power, config)
            if abs(old_cap - new_cap) < 0.05:
                found = True
            else:
                old_cap = new_cap
                power = power - power * 0.05
        sdnr_power.append(power)
        print(f"Peak Power: {peak_max_power[-1]}")
        print(f"SDNR Power: {sdnr_power[-1]}")

    os.makedirs("Cut_Figs", exist_ok=True)
    figure = plt.figure(figsize=(5, 4))
    plt.plot(x_cut_list, peak_max_power, label="Peak Power")
    plt.plot(x_cut_list, sdnr_power, label="SDNR Power")
    plt.xlabel("x_cut")
    plt.ylabel("Power")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("Cut_Figs/power.png")

    figure = plt.figure(figsize=(5, 4))
    plt.plot(x_cut_list, peak_cap, label="Peak Cap")
    plt.plot(x_cut_list, sdnr_cap, label="SDNR Cap")
    plt.xlabel("x_cut")
    plt.ylabel("Capacity")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("Cut_Figs/capacity.png")

    figure = plt.figure(figsize=(5, 4))
    plt.plot(
        x_cut_list,
        (np.array(peak_cap) - np.array(sdnr_cap)) / np.array(sdnr_cap),
    )
    plt.xlabel("x_cut")
    plt.ylabel("Capacity")
    plt.title(" (C_P - C_S )/ C_S")
    plt.grid()
    plt.tight_layout()
    plt.savefig("Cut_Figs/capacity_diff.png")

    figure = plt.figure(figsize=(5, 4))
    plt.plot(
        x_cut_list,
        (np.array(peak_max_power) - np.array(sdnr_power)) / np.array(sdnr_power),
    )
    plt.xlabel("x_cut")
    plt.ylabel("Power")
    plt.title(" (P_P - P_S )/ P_S")
    plt.grid()
    plt.tight_layout()
    plt.savefig("Cut_Figs/power_diff.png")

    print("Results saved in Cut_Figs folder")


if __name__ == "__main__":
    main()
