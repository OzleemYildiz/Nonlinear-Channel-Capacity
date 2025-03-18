from utils import (
    read_config,
    plot_interference,
    get_interference_alphabet_x_y,
    interference_dependent_snr,
    plot_res,
    plot_R1_R2_curve,
    plot_R1_vs_change,
    get_interference_alphabet_x_y_complex,
    get_regime_class_interference,
    loss,
    plot_R1_R2_change,
)
import numpy as np
from gaussian_capacity import (
    gaussian_interference_capacity,
    gaus_interference_R1_R2_curve,
    agc_gaussian_capacity_interference,
)
import os
from scipy import io
from gd import (
    gradient_descent_on_interference,
    get_fixed_interferer,
)
import time
from nonlinearity_utils import (
    get_derivative_of_nonlinear_fn,
    Hardware_Nonlinear_and_Noise,
)
import torch
from utils_interference import (
    define_save_location,
    change_parameters_range,
    get_run_parameters,
    get_linear_interference_capacity,
    get_interferer_capacity,
    get_tdm_capacity_with_optimized_dist,
    get_linear_approximation_capacity,
    get_updated_params_for_hd,
    fix_config_for_hd,
    get_int_regime,
)

# from memory_profiler import profile


def main():
    # First and Third regime implementation
    # ----System Model--- Z channel
    # Y1 = Phi(X1 + AX2 +N11)+ N12
    # Y2 = Phi(X2  + N21)+ N22

    st = time.time()
    config = read_config()

    save_location, config, str_print = define_save_location(config)
    print(str_print)

    # os.makedirs(save_location, exist_ok=True) # Now I add string at the end

    print("Save Location: ", save_location)

    # cap_RX1_no_int_no_nonlinearity = []
    # cap_RX2_no_nonlinearity = []
    cap_gaus_RX1 = []
    cap_gaus_RX2 = []

    change_range, config = change_parameters_range(config)

    if config["x2_fixed"]:
        res_change = {
            "Linear_TIN": [],
            "Gaussian_TIN": [],
            "Linear_KI": [],
            "Gaussian_KI": [],
        }
        if config["gd_active"]:
            res_change["Learned_KI"] = []
            res_change["Learned_TIN"] = []
    else:
        res_change = {"R1": {}, "R2": {}}
        res_change["R1"] = {
            "Linear": [],
            "Gaussian": [],
            "Linear_Approx_KI": [],
            "Linear_Approx_TIN": [],
        }
        res_change["R2"] = {"Linear": [], "Gaussian": []}
        if config["gd_active"]:
            res_change["R1"]["Learned"] = []
            res_change["R2"]["Learned"] = []

    if config["regime"] == 3:
        if config["x2_fixed"]:
            res_change["Linear_Approx_KI"] = []
            res_change["Linear_Approx_TIN"] = []

    old_config_title = config["title"]

    _, _, _, _, _, res_str, _ = get_run_parameters(config, change_range[0])

    save_location = save_location + res_str + "/"
    os.makedirs(save_location, exist_ok=True)

    mul_fac = []

    for ind, chng in enumerate(change_range):
        power1, power2, int_ratio, tanh_factor, tanh_factor2, res_str, res_str_run = (
            get_run_parameters(config, chng)
        )

        if config["hardware_params_active"]:
            power1, power2, config = get_updated_params_for_hd(config, power1, power2)
        else:
            config["multiplying_factor"] = 1

        mul_fac.append(config["multiplying_factor"])

        # Update Title in Config for the case of Interference
        config["title"] = config["title"] + "" + res_str

        res = {"R1": {}, "R2": {}}
        linear_tin, linear_ki = get_linear_interference_capacity(
            power1, power2, int_ratio, config
        )
        res["R1"]["Linear_TIN"] = linear_tin
        res["R1"]["Linear_KI"] = linear_ki

        linear_R2 = get_interferer_capacity(config, power2)

        res["R2"]["Linear_TIN"] = linear_R2
        res["R2"]["Linear_KI"] = linear_R2
        if config["x2_fixed"]:
            res_change["Linear_TIN"].append(linear_tin)
            res_change["Linear_KI"].append(linear_ki)
        else:
            res_change["R1"]["Linear"].append(linear_tin)
            res_change["R2"]["Linear"].append(linear_R2)

        update_save_location = save_location + config["change"] + "=" + str(chng) + "/"
        config["save_location"] = update_save_location

        os.makedirs(update_save_location, exist_ok=True)
        print("----------", str(config["change"]), ":", chng, "-----------")
        regime_RX1, regime_RX2 = get_int_regime(
            config, power1, power2, int_ratio, tanh_factor, tanh_factor2
        )

        if config["x2_fixed"]:
            save_loc_rx2 = update_save_location + "/pdf_x_RX2_opt.png"
            pdf_x_RX2 = get_fixed_interferer(
                config, regime_RX2, config["x2_type"], save_loc_rx2
            )
        else:
            pdf_x_RX2 = None

        if config["tdm_active"]:
            res_tdm = get_tdm_capacity_with_optimized_dist(
                regime_RX1,
                regime_RX2,
                config,
                pdf_x_RX2,
                update_save_location,
                save_active=True,
            )
        else:
            res_tdm = None

            # Approximated Capacity -
        if config["regime"] == 3:
            if config["x2_fixed"]:

                approx_cap_ki, approx_cap_tin = get_linear_approximation_capacity(
                    regime_RX2, config, power1, pdf_x_RX2
                )
                res_change["Linear_Approx_KI"].append(approx_cap_ki)
                res_change["Linear_Approx_TIN"].append(approx_cap_tin)
            else:
                gaus_pdf_x_RX2 = get_fixed_interferer(
                    config,
                    regime_RX2,
                    0,
                )  # give gaussian - not fixed
                approx_cap_ki, approx_cap_tin = get_linear_approximation_capacity(
                    regime_RX2, config, power1, gaus_pdf_x_RX2
                )
                res_change["R1"]["Linear_Approx_KI"].append(approx_cap_ki)
                res_change["R1"]["Linear_Approx_TIN"].append(approx_cap_tin)

        # --- Gaussian Capacity
        res_gaus = {}
        cap1_g, cap2_g = gaussian_interference_capacity(
            reg_RX1=regime_RX1,
            reg_RX2=regime_RX2,
            int_ratio=int_ratio,
            tin_active=True,  # First, we apply tin
            pdf_x_RX2=pdf_x_RX2,
        )
        if config["x2_fixed"]:
            res["R1"]["Gaussian_TIN"] = cap1_g
            res["R2"]["Gaussian_TIN"] = cap2_g
            res_change["Gaussian_TIN"].append(cap1_g)

            cap1_g, cap2_g = gaussian_interference_capacity(
                reg_RX1=regime_RX1,
                reg_RX2=regime_RX2,
                int_ratio=int_ratio,
                tin_active=False,  # Then, we apply ki
                pdf_x_RX2=pdf_x_RX2,
            )
            res["R1"]["Gaussian_KI"] = cap1_g
            res["R2"]["Gaussian_KI"] = cap2_g

            res_change["Gaussian_KI"].append(cap1_g)
        else:
            res["R1"]["Gaussian"] = cap1_g
            res["R2"]["Gaussian"] = cap2_g
            res_change["R1"]["Gaussian"].append(cap1_g)
            res_change["R2"]["Gaussian"].append(cap2_g)

        res_gaus["Gaussian"] = [cap1_g, cap2_g]
        # cap1_agc, cap2_agc = agc_gaussian_capacity_interference(config, power)
        # res_gaus["AGC"] = [cap1_agc, cap2_agc]
        # breakpoint()

        # FIXME: This takes long time
        # cap1, cap2 = gaus_interference_R1_R2_curve(config, power1, power2)

        # res["R1"]["Gaussian"] = cap1
        # res["R2"]["Gaussian"] = cap2
        if config["gd_active"]:
            if config["x2_fixed"]:
                lambda_sweep = [
                    1
                ]  # There will be only one lambda solution since x2 is fixed

            else:
                lambda_sweep = np.linspace(0.01, 0.99, config["n_lmbd"])

            (
                max_sum_cap,
                max_pdf_x_RX1,
                max_pdf_x_RX2,
                max_cap_RX1,
                max_cap_RX2,
                save_opt_sum_capacity,
            ) = gradient_descent_on_interference(
                config=config,
                reg_RX1=regime_RX1,
                reg_RX2=regime_RX2,
                lambda_sweep=lambda_sweep,
                int_ratio=int_ratio,
                tin_active=True,  # First, we apply tin
                pdf_x_RX2=pdf_x_RX2,
            )

            if config["x2_fixed"]:
                res["R1"]["Learned_TIN"] = max_cap_RX1
                res["R2"]["Learned_TIN"] = max_cap_RX2
                # if max_cap_RX1 > res_change["Linear_TIN"][-1]:
                #     breakpoint()
                res_change["Learned_TIN"].append(max_cap_RX1)
                res_pdf_tin = {
                    "RX1_tin": max_pdf_x_RX1,
                    "RX2_tin": max_pdf_x_RX2,
                }
                res_alph_tin = {
                    "RX1_tin": regime_RX1.alphabet_x
                    / 10 ** (config["multiplying_factor"] / 2),
                    "RX2_tin": regime_RX2.alphabet_x
                    / 10 ** (config["multiplying_factor"] / 2),
                }

                (
                    max_sum_cap2,
                    max_pdf_x_RX1,
                    max_pdf_x_RX2,
                    max_cap_RX1,
                    max_cap_RX2,
                    save_opt_sum_capacity2,
                ) = gradient_descent_on_interference(
                    config=config,
                    reg_RX1=regime_RX1,
                    reg_RX2=regime_RX2,
                    lambda_sweep=lambda_sweep,
                    int_ratio=int_ratio,
                    tin_active=False,  # Then, we apply ki
                    pdf_x_RX2=pdf_x_RX2,
                )
                res["R1"]["Learned_KI"] = max_cap_RX1
                res["R2"]["Learned_KI"] = max_cap_RX2
                res_change["Learned_KI"].append(max_cap_RX1)
                res_pdf_ki = {
                    "RX1_ki": max_pdf_x_RX1,
                    "RX2_ki": max_pdf_x_RX2,
                }
                res_alph_ki = {
                    "RX1_ki": regime_RX1.alphabet_x
                    / 10 ** (config["multiplying_factor"] / 2),
                    "RX2_ki": regime_RX2.alphabet_x
                    / 10 ** (config["multiplying_factor"] / 2),
                }
                res_pdf = {**res_pdf_tin, **res_pdf_ki}
                res_alph = {**res_alph_tin, **res_alph_ki}
                res_opt = {"tin": save_opt_sum_capacity, "ki": save_opt_sum_capacity2}
            else:
                res["R1"]["Learned"] = max_cap_RX1
                res["R2"]["Learned"] = max_cap_RX2

                res_change["R1"]["Learned"].append(max_cap_RX1)
                res_change["R2"]["Learned"].append(max_cap_RX2)
                res_pdf = {
                    "RX1": max_pdf_x_RX1,
                    "RX2": max_pdf_x_RX2,
                }
                res_alph = {
                    "RX1": regime_RX1.alphabet_x
                    / 10 ** (config["multiplying_factor"] / 2),
                    "RX2": regime_RX2.alphabet_x
                    / 10 ** (config["multiplying_factor"] / 2),
                }
                res_opt = {"opt": save_opt_sum_capacity}

            io.savemat(
                update_save_location + "pdf.mat",
                res_pdf,
            )
            io.savemat(
                update_save_location + "alph.mat",
                res_alph,
            )

            plot_res(
                res_opt,
                res_pdf,
                res_alph,
                update_save_location,
                lambda_sweep,
                res_str + res_str_run,
            )

        plot_R1_R2_curve(
            res,
            power1,
            power2,
            update_save_location,
            config=config,
            res_gaus=res_gaus,
            res_tdm=res_tdm,
            res_str=res_str,
        )

        io.savemat(
            update_save_location + "res_RX1.mat",
            res["R1"],
        )
        io.savemat(
            update_save_location + "res_RX2.mat",
            res["R2"],
        )

        if config["complex"]:
            del (
                # real_x1,
                # imag_x1,
                # real_y1,
                # imag_y1,
                # real_x2,
                # imag_x2,
                # real_y2,
                # imag_y2,
                regime_RX1,
                regime_RX2,
            )
        else:
            del (
                # alphabet_x_RX1,
                # alphabet_y_RX1,
                # alphabet_x_RX2,
                # alphabet_y_RX2,
                regime_RX1,
                regime_RX2,
            )
        if config["hardware_params_active"]:
            config = fix_config_for_hd(config)

    if config["x2_fixed"]:

        plot_R1_vs_change(res_change, change_range, config, save_location, res_str)
        res_change["change_range"] = change_range
        res_change["change_over"] = config["change"]
        io.savemat(
            save_location
            + "res_change-"
            + str(config["change"])
            + "_"
            + res_str
            + ".mat",
            res_change,
        )
    else:
        if config["gd_active"]:
            res_change = plot_R1_R2_change(
                res_change, change_range, config, save_location, res_str, lambda_sweep
            )
        else:
            res_change = plot_R1_R2_change(
                res_change, change_range, config, save_location, res_str
            )

        res_change["change_range"] = change_range
        res_change["change_over"] = config["change"]

        io.savemat(
            save_location
            + "res_change-"
            + str(config["change"])
            + "_"
            + res_str
            + ".mat",
            res_change,
        )
    print("Time taken: ", time.time() - st)
    # res = {
    #     "Capacity_without_Interference_Nonlinearity_RX1": cap_RX1_no_int_no_nonlinearity,
    #     "Capacity_without_Nonlinearity_RX2": cap_RX2_no_nonlinearity,
    #     # "Capacity_Gaussian_RX1": ,
    #     # "Capacity_Gaussian_RX2": cap_gaus_RX2,
    # }
    # io.savemat(save_location + "/res.mat", res)
    io.savemat(save_location + "/config.mat", config)

    # plot_interference(res, config, save_location)

    print("Saved in ", save_location)


if __name__ == "__main__":

    main()
