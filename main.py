import yaml
import argparse
import os
import torch
import matplotlib.pyplot as plt
from gaussian_capacity import (
    gaussian_capacity,
    gaussian_with_l1_norm,
    find_best_gaussian,
)
from gd import gd_capacity, gd_on_alphabet_capacity
from utils import (
    get_alphabet_x_y,
    plot_vs_change,
    plot_pdf_vs_change,
    project_pdf,
    return_regime_class,
    regime_dependent_snr,
    read_config,
    loss,
    plot_R1_R2_curve,
)
import numpy as np
from bounds import (
    bounds_l1_norm,
    upper_bound_tarokh,
    lower_bound_with_sdnr,
    lower_bound_tarokh,
    lower_bound_tarokh_third_regime,
    lower_bound_tarokh_third_regime_with_pw,
    sdnr_bound_regime_1_tarokh_ref7,
    updated_sdnr_bound_regime_1_tarokh_ref7,
    sdnr_new,
    sdnr_new_rayleigh,
    sdnr_new_with_erf,
    sundeep_upper_bound_third_regime,
    upper_bound_tarokh_third_regime,
    lower_bound_by_mmse,
    lower_bound_by_mmse_with_truncated_gaussian,
    reg_mmse_bound_numerical,
    upper_bound_peak,
    bound_backtracing_check,
    sdnr_new_with_erf_nopowchange,
    lower_bound_by_mmse_correlation,
    lower_bound_by_mmse_correlation_numerical,
)
from scipy import io
import time
from blahut_arimoto_capacity import apply_blahut_arimoto


def define_save_location(config):
    save_location = config["output_dir"]
    if config["time_division_active"]:
        save_location = save_location + "-TDM"
        if config["power_change_active"]:
            save_location = save_location + "-PB"

    save_location = save_location + (
        "/"
        + config["cons_str"]
        + "_phi="
        + str(config["nonlinearity"])
        + "_regime="
        + str(config["regime"])
        + "_tdm="
        + str(config["time_division_active"])
    )
    if config["time_division_active"]:
        save_location = (
            save_location
            + "_min_pow="
            + str(config["min_power_cons"])
            + "_power_change_active="
            + str(config["power_change_active"])
        )

    if config["regime"] == 1:
        save_location = save_location + "_sigma2=" + str(config["sigma_2"])
    elif config["regime"] == 2:
        save_location = save_location + "_sigma1=" + str(config["sigma_1"])
    elif config["regime"] == 3:
        save_location = (
            save_location
            + "_sigma1="
            + str(config["sigma_1"])
            + "_sigma2="
            + str(config["sigma_2"])
        )

    if config["gd_active"]:
        save_location = save_location + "_gd=" + str(config["gd_active"])
        if config["gd_initial_ba"]:
            save_location = save_location + "_initial_ba"
    if config["gd_alphabet_active"]:
        save_location = (
            save_location + "_gd_alphabet_" + str(config["gd_alphabet_active"])
        )
    if config["nonlinearity"] == 5:
        save_location = (
            save_location
            + "_clipx="
            + str(config["clipping_limit_x"])
            + "_clipy="
            + str(config["clipping_limit_y"])
        )
    elif config["nonlinearity"] == 3:
        save_location = (
            save_location
            + "_"
            + str(config["tanh_factor"])
            + "tanh"
            + "x/"
            + str(config["tanh_factor"])
        )
    save_location = save_location + "/"
    os.makedirs(
        save_location,
        exist_ok=True,
    )
    return save_location


def main():
    config = read_config()

    print(
        "**** AWGN Channel with Nonlinearity: ",
        config["nonlinearity"],
        "for Regime: ",
        config["regime"],
        "****",
    )

    # Since we are looking at TDM - SNR needs to be fixed so we only take one SNR value
    if config["time_division_active"]:
        if config["n_snr"] > 1:
            print("WARNING!! Time Division Active: Only minimum SNR will be considered")
            config["n_snr"] = 1

    snr_change, noise_power = regime_dependent_snr(config)

    save_location = define_save_location(config)
    print("Saving in: " + save_location)

    capacity_gaussian = []
    power_gaussian = []
    capacity_learned = []
    capacity_learned_over_alphabet = []
    capacity_ruth = []
    capacity_ba = []
    calc_logsnr = []

    map_pdf = {}
    map_pdf_ba = {}  # TODO: Change this too
    alphabet_x = []
    up_tarokh = []
    low_sdnr = []
    low_tarokh_third = []
    sdnr_tarokh_low = []
    my_new_bound = []
    sundeep_upper = []
    tarokh_upper = []
    linear_mmse_bound = []
    mmse_bound = []
    mmse_minimum = []
    mmse_correlation = []
    up_peak = []

    if config["time_division_active"]:
        tau_list = np.linspace(0.01, 0.99, config["n_time_division"])
        # Fill this if and only time division is active
        rate_1_tau_gaussian = []
        rate_2_tau_gaussian = []
        rate_1_tau_learned = []
        rate_2_tau_learned = []
    else:
        tau_list = np.array([1])
    power_change = []
    for snr in snr_change:
        start = time.time()
        print("-------SNR in dB:", snr, "--------")
        power = (10 ** (snr / 10)) * noise_power
        power_change.append(power)
        res_tau = {"R1": {}, "R2": {}}

        for tau in tau_list:
            print("----------Time Division: ", tau, "----------")
            if config["power_change_active"]:
                tau_power_list = [power / tau, power / (1 - tau)]
            else:
                tau_power_list = [power]  # No power boost
            for ind, tau_power in enumerate(tau_power_list):
                power = tau_power
                if config["power_change_active"]:
                    print("Power Boost: ", power, " for User ", ind + 1)
                else:
                    print("Power Kept Constant: ", power)
                if ind == 0:  # keeping record of only tau results for demonstration
                    # 1-tau is for R1 and R2 curve
                    calc_logsnr.append(np.log(1 + power / (noise_power)) / 2)
                print("log(1+SNR)/2 :", calc_logsnr[-1])

                alphabet_x, alphabet_y, max_x, max_y = get_alphabet_x_y(config, power)

                regime_class = return_regime_class(
                    config, alphabet_x, alphabet_y, power
                )

                # FIXME: Currently, the bounds are not calculated for TDM
                if config["bound_active"] and ind == 0:
                    if config["regime"] == 1 and config["cons_type"] == 1:
                        up_tarokh.append(upper_bound_tarokh(power, config))
                        if config["nonlinearity"] == 5:

                            my_new_bound = bound_backtracing_check(
                                my_new_bound,
                                sdnr_new_with_erf_nopowchange(power, config),
                            )

                        # mmse_correlation = bound_backtracing_check(
                        #     mmse_correlation,
                        #     lower_bound_by_mmse_correlation(power, config),
                        # )

                        # linear_mmse_bound = bound_backtracing_check(
                        #     linear_mmse_bound, lower_bound_by_mmse(power, config)
                        # )
                        mmse_minimum = bound_backtracing_check(
                            mmse_minimum, reg_mmse_bound_numerical(power, config)
                        )
                    if (
                        config["regime"] == 2 and config["cons_type"] == 1
                    ):  # average power
                        # up_tarokh.append((calc_logsnr[-1]))
                        low_sdnr.append(lower_bound_with_sdnr(power, config))
                    if config["regime"] == 3:
                        # sundeep_upper.append(
                        #     sundeep_upper_bound_third_regime(power, config)
                        # )
                        if config["nonlinearity"] != 5:
                            tarokh_upper.append(
                                upper_bound_tarokh_third_regime(power, config)
                            )
                    if config["nonlinearity"] == 5:
                        up_peak.append(upper_bound_peak(power, config))

                # Gaussian Capacity
                # cap_g = gaussian_capacity(regime_class)
                # power_g, cap_g = find_best_gaussian(regime_class)
                if ind == 0:  # keeping record of only tau results for demonstration
                    capacity_gaussian = bound_backtracing_check(
                        capacity_gaussian, gaussian_capacity(regime_class, power)
                    )
                    # capacity_gaussian.append(cap_g)
                    # power_gaussian.append(power_g)

                if config["ba_active"] and config["cons_type"] == 0:
                    (cap, input_dist) = apply_blahut_arimoto(regime_class, config)
                    loss_g = loss(
                        torch.tensor(input_dist).to(torch.float32), regime_class
                    )
                    if ind == 0:  # keeping record of only tau results for demonstration
                        capacity_ba.append(-loss_g)
                    if not config["time_division_active"]:
                        map_pdf_ba[str(snr)] = [
                            input_dist,
                            regime_class.alphabet_x.numpy(),
                        ]
                    else:
                        map_pdf_ba[str(tau)] = [
                            input_dist,
                            regime_class.alphabet_x.numpy(),
                        ]

                # If constraint type is 2, calculate gaussian with optimized snr  -- Ruth's paper
                # TODO: Might not be working
                if config["cons_type"] == 2:
                    cap_r = gaussian_with_l1_norm(alphabet_x, alphabet_y, power, config)
                    capacity_ruth.append(cap_r)

                # Gradient Descent
                if config["gd_active"]:
                    cap_learned, max_pdf_x, max_alphabet_x, opt_capacity = gd_capacity(
                        config, power, regime_class
                    )

                    # FIXME: Messy
                    plt.figure()
                    plt.plot(opt_capacity)
                    plt.xlabel("Iterations")
                    plt.grid()
                    plt.savefig(save_location + "/opt_capacity.png")
                    plt.close()
                    io.savemat(
                        save_location + "/opt_capacity.mat",
                        {"opt_capacity": opt_capacity},
                    )

                    if ind == 0:  # keeping record of only tau results for demonstration
                        capacity_learned.append(cap_learned)
                    max_pdf_x = project_pdf(
                        max_pdf_x, config["cons_type"], max_alphabet_x, power
                    )
                    if config["time_division_active"]:
                        if config["power_change_active"]:
                            map_pdf[
                                "Chng" + str(int(tau * 100)) + "ind=" + str(ind)
                            ] = [
                                max_pdf_x.detach().numpy(),
                                max_alphabet_x.detach().numpy(),
                            ]
                        else:
                            map_pdf["Chng" + str(int(tau * 100))] = [
                                max_pdf_x.detach().numpy(),
                                max_alphabet_x.detach().numpy(),
                            ]
                    else:
                        map_pdf["Chng" + str(int(snr * 100))] = [
                            max_pdf_x.detach().numpy(),
                            max_alphabet_x.detach().numpy(),
                        ]
                # Gradient Descent on Alphabet X for Peak Power Constraint
                # TODO: Not working for TDM
                if config["gd_alphabet_active"] and config["cons_type"] == 0:
                    cap_learned, max_pdf_x, max_alphabet_x = gd_on_alphabet_capacity(
                        max_x, config, power, regime_class
                    )
                    capacity_learned_over_alphabet.append(cap_learned)
                    map_pdf["Chng" + str(int(snr * 100))] = [
                        max_pdf_x.detach().numpy(),
                        max_alphabet_x.detach().numpy(),
                    ]

                del regime_class, alphabet_x, alphabet_y
                end = time.time()

                # Time Division Active Calculation
                if config["time_division_active"]:

                    if config["power_change_active"]:
                        if ind == 0:
                            rate_1_tau_gaussian.append(tau * cap_g)
                            rate_1_tau_learned.append(tau * cap_learned)
                        if ind == 1:
                            rate_2_tau_gaussian.append((1 - tau) * cap_g)
                            rate_2_tau_learned.append((1 - tau) * cap_learned)
                    else:
                        # Since we only have one SNR value that we learn the distribution for, we can break here
                        rate_1_tau_gaussian.append((tau_list * cap_g.numpy()))
                        rate_1_tau_gaussian = rate_1_tau_gaussian[0]
                        rate_2_tau_gaussian.append((1 - tau_list) * cap_g.numpy())
                        rate_2_tau_gaussian = rate_2_tau_gaussian[0]
                        rate_1_tau_learned.append(tau_list * cap_learned)
                        rate_1_tau_learned = rate_1_tau_learned[0]
                        rate_2_tau_learned.append((1 - tau_list) * cap_learned)
                        rate_2_tau_learned = rate_2_tau_learned[0]

            if config["time_division_active"] and not config["power_change_active"]:
                break

        if config["time_division_active"]:
            res_tau["R1"]["Gaussian"] = rate_1_tau_gaussian
            res_tau["R2"]["Gaussian"] = rate_2_tau_gaussian
            res_tau["R1"]["Learned"] = rate_1_tau_learned
            res_tau["R2"]["Learned"] = rate_2_tau_learned

            pow_original = (10 ** (snr / 10)) * noise_power
            plot_R1_R2_curve(
                res_tau, pow_original, config=config, save_location=save_location
            )
            io.savemat(save_location + "/res_tau.mat", res_tau)

        # print("Time taken for SNR: ", snr, " is ", end - start)

    low_tarokh = None
    if (
        config["bound_active"]
        and config["regime"] == 1
        and config["cons_type"] == 1
        and config["nonlinearity"] != 5
    ):
        low_tarokh, snr_tarokh = lower_bound_tarokh(config)
        # max handle
        if max(snr_change) > np.max(snr_tarokh[:-1]):
            snr_tarokh[-1] = max(snr_change)
            if min(snr_change) > np.max(snr_tarokh[:-1]):
                snr_tarokh = [min(snr_change), snr_tarokh[-1]]
                low_tarokh = [low_tarokh[-1], low_tarokh[-1]]
        else:
            snr_tarokh = snr_tarokh[:-1]
            low_tarokh = low_tarokh[:-1]

        ind = np.where(np.array(snr_tarokh) >= min(snr_change))
        low_tarokh = np.array(low_tarokh)[ind]
        snr_tarokh = np.array(snr_tarokh)[ind]
        low_tarokh = {"SNR": snr_tarokh, "Lower_Bound": low_tarokh}
        io.savemat(
            define_save_location(config) + "/low_tarokh.mat",
            low_tarokh,
        )

    res = {
        "Gaussian_Capacity": capacity_gaussian,
        "Capacity_without_Nonlinearity": calc_logsnr,
    }
    if config["gd_active"]:
        res["Learned_Capacity"] = capacity_learned
    if config["gd_alphabet_active"] and config["cons_type"] == 0:
        res["Learned_Capacity_Over_Alphabet"] = capacity_learned_over_alphabet
    if config["bound_active"]:
        if config["regime"] == 1 and config["cons_type"] == 1:
            res["Upper_Bound_by_Tarokh"] = up_tarokh
            if config["nonlinearity"] == 5:
                # res["Lower_Bound_by_SDNR"] = sdnr_tarokh_low
                res["SDNR_with_Gaussian"] = my_new_bound
            # res["MMSE_Bound"] = mmse_bound
            res["Linear_MMSE_Bound"] = linear_mmse_bound
            res["MMSE_Bound_Regular"] = mmse_minimum
            res["MMSE_Bound_Correlation"] = mmse_correlation
            # res["MMSE_Bound_Correlation_Numerical"] = mmse_correlation_numerical
        if config["regime"] == 2 and config["cons_type"] == 1:  # average power bound
            res["Lower_Bound_with_SDNR"] = low_sdnr
        if config["regime"] == 3 and config["cons_type"] == 1:
            # res["Lower_Bound_by_Tarokh"] = low_tarokh_third
            if config["nonlinearity"] != 5:
                low, snr_tarokh = lower_bound_tarokh_third_regime(config)
                low_tarokh = {"SNR": snr_tarokh, "Lower_Bound": low}
                io.savemat(
                    define_save_location(config) + "/low_tarokh.mat",
                    low_tarokh,
                )
            if config["nonlinearity"] != 5:
                res["Tarokh Upper Bound"] = tarokh_upper
        if config["nonlinearity"] == 5:
            res["Upper_Bound_Peak"] = up_peak

    if config["cons_type"] == 2:
        res["Gaussian_Capacity_with_L1_Norm"] = capacity_ruth
    if config["ba_active"] and config["cons_type"] == 0:
        res["Blahut_Arimoto_Capacity"] = capacity_ba

    if not config["time_division_active"]:
        # If it's active, the goal is to plot R1 and R2 curves
        plot_vs_change(
            snr_change, res, config, save_location=save_location, low_tarokh=low_tarokh
        )

        if config["gd_active"] or (
            config["gd_alphabet_active"] and config["cons_type"] == 0
        ):

            plot_pdf_vs_change(
                map_pdf,
                snr_change,
                config,
                save_location=save_location,
                file_name="pdf_snr_gd.png",
            )
        if config["ba_active"] and config["cons_type"] == 0:
            plot_pdf_vs_change(
                map_pdf_ba,
                snr_change,
                config,
                save_location=save_location,
                file_name="pdf_snr_ba.png",
            )
        io.savemat(
            save_location + "/pdf.mat",
            map_pdf,
        )

    else:
        # If it's active, the goal is to plot R1 and R2 curves
        if not (not config["power_change_active"] and config["time_division_active"]):
            plot_vs_change(
                tau_list,
                res,
                config,
                save_location=save_location,
                low_tarokh=low_tarokh,
            )

        if config["gd_active"] or (
            config["gd_alphabet_active"] and config["cons_type"] == 0
        ):
            io.savemat(
                save_location + "/pdf.mat",
                map_pdf,
            )
            plot_pdf_vs_change(
                map_pdf,
                tau_list,
                config,
                save_location=save_location,
                file_name="pdf_snr_gd.png",
            )
        if config["ba_active"] and config["cons_type"] == 0:
            plot_pdf_vs_change(
                map_pdf_ba,
                tau_list,
                config,
                save_location=save_location,
                file_name="pdf_snr_ba.png",
            )

    res["Power_Gaussian"] = power_gaussian
    res["Power_Change"] = power_change
    io.savemat(
        save_location + "/res.mat",
        res,
    )
    io.savemat(
        save_location + "/config.mat",
        config,
    )

    print("Results saved at: ", save_location)


if __name__ == "__main__":
    main()
