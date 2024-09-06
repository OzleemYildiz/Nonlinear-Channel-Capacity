import yaml
import argparse
import os
import matplotlib.pyplot as plt
from gaussian_capacity import gaussian_capacity, gaussian_with_l1_norm
from gd import gd_capacity
from utils import generate_alphabet_x_y, plot_snr, plot_pdf_snr, project_pdf
import numpy as np
from bounds import bounds_l1_norm, upper_bound_tarokh, lower_bound_with_sdnr
from scipy import io


def main():
    # READ CONFIG FILE
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="arguments.yml",
        help="Configure of post processing",
    )
    args = parser.parse_args()
    config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    os.makedirs(config["output_dir"], exist_ok=True)
    # Constraint type of the system
    if config["cons_type"] == 1:
        config["cons_str"] = "Avg"
    elif config["cons_type"] == 0:
        config["cons_str"] = "Peak"
    else:
        config["cons_str"] = "First"

    print("**** AWGN Channel with Nonlinearity: ", config["nonlinearity"], "****")

    # Number of SNR points to be evaluated
    snr_change = np.linspace(
        10 * np.log10(config["min_power_cons"] / (config["sigma"] ** 2)),
        10 * np.log10(config["max_power_cons"] / (config["sigma"] ** 2)),
        config["n_snr"],
    )
    capacity_gaussian = []
    capacity_learned = []
    capacity_ruth = []

    calc_logsnr = []

    map_snr_pdf = {}
    up_tarokh = []
    low_sdnr = []

    for snr in snr_change:
        power = (10 ** (snr / 10)) * config["sigma"] ** 2
        print("-------SNR in dB:", snr, "--------")

        calc_logsnr.append(np.log(1 + power / (config["sigma"] ** 2)) / 2)
        print("log(1+SNR)/2 :", calc_logsnr[-1])

        alphabet_x, alphabet_y, max_x, max_y = generate_alphabet_x_y(config, power)

        if config["bound_active"]:
            if config["regime"] == 1:
                up_tarokh.append(upper_bound_tarokh(power, config))
            if config["regime"] == 2:
                # up_tarokh.append((calc_logsnr[-1]))
                low_sdnr.append(lower_bound_with_sdnr(power, config))

        # Gaussian Capacity
        cap_g = gaussian_capacity(alphabet_x, alphabet_y, power, config)
        capacity_gaussian.append(cap_g)

        # If constraint type is 2, calculate gaussian with optimized snr  -- Ruth's paper
        if config["cons_type"] == 2:
            cap_r = gaussian_with_l1_norm(alphabet_x, alphabet_y, power, config)
            capacity_ruth.append(cap_r)

        # Gradient Descent
        if config["gd_active"]:
            cap_learned, max_pdf_x = gd_capacity(max_x, alphabet_y, config, power)
            capacity_learned.append(cap_learned)
            max_pdf_x = project_pdf(max_pdf_x, config["cons_type"], alphabet_x, power)
            map_snr_pdf[snr] = [max_pdf_x, max_alphabet_x]
    res = {
        "Gaussian_Capacity": capacity_gaussian,
        "Capacity_without_Nonlinearity": calc_logsnr,
    }
    if config["gd_active"]:
        res["Learned_Capacity"] = capacity_learned
    if config["bound_active"]:
        if config["regime"] == 1:
            res["Upper_Bound_by_Tarokh"] = up_tarokh
        if config["regime"] == 2:
            res["Lower_Bound_with_SDNR"] = low_sdnr
    if config["cons_type"] == 2:
        res["Gaussian_Capacity_with_L1_Norm"] = capacity_ruth

    os.makedirs(
        config["output_dir"]
        + "/"
        + config["cons_str"]
        + "_nonlinearity="
        + str(config["nonlinearity"])
        + "_regime="
        + str(config["regime"])
        + "_gd_"
        + str(config["gd_active"])
        + "/",
        exist_ok=True,
    )
    io.savemat(
        config["output_dir"]
        + "/"
        + config["cons_str"]
        + "_nonlinearity="
        + str(config["nonlinearity"])
        + "_regime="
        + str(config["regime"])
        + "_gd_"
        + str(config["gd_active"])
        + "/res.mat",
        res,
    )
    plot_snr(snr_change, res, config)
    if config["gd_active"]:
        plot_pdf_snr(map_snr_pdf, snr_change, config)
        io.savemat(
            config["output_dir"]
            + "/"
            + config["cons_str"]
            + "_nonlinearity="
            + str(config["nonlinearity"])
            + "_regime="
            + str(config["regime"])
            + "_gd_"
            + str(config["gd_active"])
            + "/pdf.mat",
            {"map_snr_pdf": map_snr_pdf},
        )


if __name__ == "__main__":
    main()
