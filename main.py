import yaml
import argparse
import os
import matplotlib.pyplot as plt
from gaussian_capacity import gaussian_capacity, gaussian_with_l1_norm
from gd import gd_capacity
from utils import (
    generate_alphabet_x_y,
    plot_snr,
    plot_pdf_snr,
    project_pdf,
    return_regime_class,
    regime_dependent_snr,
    read_config,
)
import numpy as np
from bounds import bounds_l1_norm, upper_bound_tarokh, lower_bound_with_sdnr
from scipy import io
import time
from blahut_arimoto_capacity import apply_blahut_arimoto


def main():
    config = read_config()

    print("**** AWGN Channel with Nonlinearity: ", config["nonlinearity"], "****")
    snr_change, noise_power = regime_dependent_snr(config)

    capacity_gaussian = []
    capacity_learned = []
    capacity_ruth = []
    capacity_ba = []
    calc_logsnr = []

    map_snr_pdf = {}
    map_snr_pdf_ba = {}
    up_tarokh = []
    low_sdnr = []

    for snr in snr_change:
        start = time.time()

        print("-------SNR in dB:", snr, "--------")
        power = (10 ** (snr / 10)) * noise_power
        calc_logsnr.append(np.log(1 + power / (noise_power)) / 2)
        print("log(1+SNR)/2 :", calc_logsnr[-1])

        alphabet_x, alphabet_y, max_x, max_y = generate_alphabet_x_y(config, power)
        regime_class = return_regime_class(config, alphabet_x, alphabet_y, power)

        if config["bound_active"]:
            if config["regime"] == 1:
                up_tarokh.append(upper_bound_tarokh(power, config))
            if config["regime"] == 2:
                # up_tarokh.append((calc_logsnr[-1]))
                low_sdnr.append(lower_bound_with_sdnr(power, config))

        # Gaussian Capacity
        cap_g = gaussian_capacity(regime_class)
        capacity_gaussian.append(cap_g)

        # If constraint type is 2, calculate gaussian with optimized snr  -- Ruth's paper
        if config["cons_type"] == 2:
            cap_r = gaussian_with_l1_norm(alphabet_x, alphabet_y, power, config)
            capacity_ruth.append(cap_r)

        # Gradient Descent
        if config["gd_active"]:
            cap_learned, max_pdf_x, max_alphabet_x = gd_capacity(
                max_x, config, power, regime_class
            )

            capacity_learned.append(cap_learned)
            max_pdf_x = project_pdf(
                max_pdf_x, config["cons_type"], max_alphabet_x, power
            )
            map_snr_pdf[str(snr)] = [
                max_pdf_x.detach().numpy(),
                max_alphabet_x.detach().numpy(),
            ]

        if config["ba_active"] and config["cons_type"] == 0:
            cap, input_dist = apply_blahut_arimoto(regime_class, config)
            capacity_ba.append(cap)
            map_snr_pdf_ba[str(snr)] = [input_dist, regime_class.alphabet_x.numpy()]

        del regime_class, alphabet_x, alphabet_y
        end = time.time()
        # print("Time taken for SNR: ", snr, " is ", end - start)

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
    if config["ba_active"] and config["cons_type"] == 0:
        res["Blahut_Arimoto_Capacity"] = capacity_ba

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
        plot_pdf_snr(map_snr_pdf, snr_change, config, file_name="pdf_snr_gd.png")
        # io.savemat(
        #     config["output_dir"]
        #     + "/"
        #     + config["cons_str"]
        #     + "_nonlinearity="
        #     + str(config["nonlinearity"])
        #     + "_regime="
        #     + str(config["regime"])
        #     + "_gd_"
        #     + str(config["gd_active"])
        #     + "/pdf.mat",
        #     map_snr_pdf,
        # )
        # plot_pdf_snr(map_snr_pdf, snr_change, config, save_location=file_name)
    if config["ba_active"] and config["cons_type"] == 0:
        plot_pdf_snr(map_snr_pdf_ba, snr_change, config, file_name="pdf_snr_ba.png")


if __name__ == "__main__":
    main()
