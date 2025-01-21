from utils import (
    read_config,
    plot_interference,
    get_interference_alphabet_x_y,
    interference_dependent_snr,
    plot_res,
    plot_R1_R2_curve,
    plot_R1_vs_change,
    get_interference_alphabet_x_y_complex,
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
    gradient_descent_projection_with_learning_rate,
)
import time


def define_save_location(config):
    save_location = (
        config["output_dir"]
        + "/"
        + config["cons_str"]
        + "_phi="
        + str(config["nonlinearity"])
    )
    if config["nonlinearity"] == 5:
        save_location = save_location + "_clip=" + str(config["clipping_limit_x"])

    save_location = (
        save_location
        + "_regime="
        + str(config["regime"])
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
    elif config["regime"] == 3:
        save_location = (
            save_location
            + "_sigma11="
            + str(config["sigma_11"])
            + "_sigma12="
            + str(config["sigma_12"])
            + "_sigma21="
            + str(config["sigma_21"])
            + "_sigma22="
            + str(config["sigma_22"])
        )

    if config["gd_active"]:
        save_location = save_location + "_gd_" + str(config["gd_active"])
        if config["x2_fixed"]:
            save_location = save_location + "_x2_fixed"
            save_location = save_location + "_x2=" + str(config["x2_type"])

    save_location = save_location + "/"

    return save_location


def change_parameters_range(config):
    if config["change"] == "pw1":
        change_range = np.logspace(
            np.log10(config["min_power1"]),
            np.log10(config["max_power1"]),
            config["n_change"],
        )
        print("-------------Change is Power1-------------")
    elif config["change"] == "pw2":
        change_range = np.logspace(
            np.log10(config["min_power2"]),
            np.log10(config["max_power2"]),
            config["n_change"],
        )
        print("-------------Change is Power2-------------")
    elif config["change"] == "a":
        change_range = np.linspace(
            config["min_int_ratio"],
            config["max_int_ratio"],
            config["n_change"],
        )
        print("-------------Change is Int Ratio-------------")
    elif config["change"] == "k":
        if config["nonlinearity"] != 3:
            raise ValueError("Tanh Factor is only for nonlinearity 3")
        change_range = np.linspace(
            config["min_tanh_factor"],
            config["max_tanh_factor"],
            config["n_change"],
        )
        print("-------------Change is Tanh Factor for User 1-------------")
    elif config["change"] == "k2":
        if config["nonlinearity"] != 3:
            raise ValueError("Tanh Factor is only for nonlinearity 3")
        change_range = np.linspace(
            config["min_tanh_factor_2"],
            config["max_tanh_factor_2"],
            config["n_change"],
        )
        print("-------------Change is Tanh Factor for User 2-------------")
    else:
        raise ValueError("Change parameter is not defined")
    return change_range


def get_run_parameters(config, chng):
    power1 = config["min_power1"]
    power2 = config["min_power2"]
    int_ratio = config["min_int_ratio"]
    tanh_factor = config["min_tanh_factor"]
    tanh_factor2 = config["min_tanh_factor_2"]
    if config["change"] == "pw1":
        power1 = chng
        res_str = (
            "pw2="
            + str(power2)
            + "_a="
            + str(int_ratio)
            + "_k="
            + str(tanh_factor)
            + "_k2="
            + str(tanh_factor)
        )
    elif config["change"] == "pw2":
        power2 = chng
        res_str = (
            "pw1="
            + str(power1)
            + "_a="
            + str(int_ratio)
            + "_k="
            + str(tanh_factor)
            + "_k2="
            + str(tanh_factor2)
        )
    elif config["change"] == "a":
        int_ratio = chng
        res_str = (
            "pw1="
            + str(power1)
            + "_pw2="
            + str(power2)
            + "_k="
            + str(tanh_factor)
            + "_k2="
            + str(tanh_factor2)
        )
    elif config["change"] == "k":
        tanh_factor = chng
        res_str = (
            "pw1="
            + str(power1)
            + "_pw2="
            + str(power2)
            + "_k2="
            + str(tanh_factor2)
            + "_a="
            + str(int_ratio)
        )
    elif config["change"] == "k2":
        tanh_factor2 = chng
        res_str = (
            "pw1="
            + str(power1)
            + "_pw2="
            + str(power2)
            + "_k="
            + str(tanh_factor)
            + "_a="
            + str(int_ratio)
        )

    return power1, power2, int_ratio, tanh_factor, tanh_factor2, res_str


def get_linear_interference_capacity(power1, power2, int_ratio, config):
    # This is X2 fixed results --- X1's tin and ki results

    if config["regime"] == 3:
        linear_ki = (
            1
            / 2
            * np.log(1 + power1 / (config["sigma_12"] ** 2 + config["sigma_11"] ** 2))
        )
        linear_tin = (
            1
            / 2
            * np.log(
                1
                + power1
                / (
                    int_ratio**2 * power2
                    + config["sigma_12"] ** 2
                    + config["sigma_11"] ** 2
                )
            )
        )
    elif config["regime"] == 1:
        linear_ki = 1 / 2 * np.log(1 + power1 / config["sigma_12"] ** 2)
        linear_tin = (
            1
            / 2
            * np.log(1 + power1 / (int_ratio**2 * power2 + config["sigma_12"] ** 2))
        )
    else:
        raise ValueError("Regime not defined")
    return linear_tin, linear_ki


def main():
    # First and Third regime implementation
    # ----System Model--- Z channel
    # Y1 = Phi(X1 + AX2 +N11)+ N12
    # Y2 = Phi(X2  + N21)+ N22

    st = time.time()
    config = read_config()
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

    # cap_RX1_no_int_no_nonlinearity = []
    # cap_RX2_no_nonlinearity = []
    cap_gaus_RX1 = []
    cap_gaus_RX2 = []

    change_range = change_parameters_range(config)
    res_change = {
        "Learned_TIN": [],
        "Linear_TIN": [],
        "Gaussian_TIN": [],
        "Learned_KI": [],
        "Linear_KI": [],
        "Gaussian_KI": [],
    }

    for ind, chng in enumerate(change_range):
        power1, power2, int_ratio, tanh_factor, tanh_factor2, res_str = (
            get_run_parameters(config, chng)
        )
        res = {"R1": {}, "R2": {}}
        linear_tin, linear_ki = get_linear_interference_capacity(
            power1, power2, int_ratio, config
        )
        res_change["Linear_TIN"].append(linear_tin)
        res_change["Linear_KI"].append(linear_ki)

        update_save_location = (
            save_location
            + "power1="
            + str(round(power1, 5))
            + "/"
            + "power2="
            + str(round(power2, 5))
            + "/"
            + "int_ratio="
            + str(round(int_ratio, 5))
            + "/"
            + "tanh_factor="
            + str(round(tanh_factor, 5))
            + "/"
            + "tanh_factor2="
            + str(round(tanh_factor2, 5))
            + "/"
        )

        os.makedirs(update_save_location, exist_ok=True)
        print("----------", str(config["change"]), ":", chng, "-----------")
        if config["complex"]:
            real_x1, imag_x1, real_y_1, imag_y1, real_x2, imag_x2, real_y2, imag_y2 = (
                get_interference_alphabet_x_y_complex(
                    config, power1, power2, int_ratio, tanh_factor, tanh_factor2
                )
            )
            
            breakpoint()
            # !!! Complex Regimes implement here - functions should take regime as input!!!!

        else:
            alphabet_x_RX1, alphabet_y_RX1, alphabet_x_RX2, alphabet_y_RX2 = (
                get_interference_alphabet_x_y(
                    config, power1, power2, int_ratio, tanh_factor, tanh_factor2
                )
            )

        # ---
        res_gaus = {}
        config["x1_update_scheme"] = 0  # First, we apply tin
        cap1_g, cap2_g = gaussian_interference_capacity(
            power1,
            power2,
            config,
            alphabet_x_RX1,
            alphabet_y_RX1,
            alphabet_x_RX2,
            alphabet_y_RX2,
            tanh_factor,
            tanh_factor2,
            int_ratio,
        )
        if config["x2_fixed"]:
            res["R1"]["Gaussian_TIN"] = cap1_g
            res["R2"]["Gaussian_TIN"] = cap2_g
            res_change["Gaussian_TIN"].append(cap1_g)
            config["x1_update_scheme"] = 1  # Then, we apply ki
            cap1_g, cap2_g = gaussian_interference_capacity(
                power1,
                power2,
                config,
                alphabet_x_RX1,
                alphabet_y_RX1,
                alphabet_x_RX2,
                alphabet_y_RX2,
                tanh_factor,
                tanh_factor2,
                int_ratio,
            )
            res["R1"]["Gaussian_KI"] = cap1_g
            res["R2"]["Gaussian_KI"] = cap2_g

            res_change["Gaussian_KI"].append(cap1_g)
        else:
            res["R1"]["Gaussian"] = cap1_g
            res["R2"]["Gaussian"] = cap2_g

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
                config["x1_update_scheme"] = 0  # First, we apply tin
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
                config,
                power1,
                power2,
                lambda_sweep,
                tanh_factor,
                tanh_factor2,
                int_ratio,
            )
            if config["x2_fixed"]:
                res["R1"]["Learned_TIN"] = max_cap_RX1
                # if max_cap_RX1 > res_change["Linear_TIN"][-1]:
                #     breakpoint()
                res_change["Learned_TIN"].append(max_cap_RX1)
                res_pdf_tin = {
                    "RX1_tin": max_pdf_x_RX1,
                    "RX2_tin": max_pdf_x_RX2,
                }
                res_alph_tin = {
                    "RX1_tin": alphabet_x_RX1,
                    "RX2_tin": alphabet_x_RX2,
                }
                if max_cap_RX1 > linear_tin:
                    breakpoint()

                config["x1_update_scheme"] = 1  # Then, we apply ki
                (
                    max_sum_cap2,
                    max_pdf_x_RX1,
                    max_pdf_x_RX2,
                    max_cap_RX1,
                    max_cap_RX2,
                    save_opt_sum_capacity2,
                ) = gradient_descent_on_interference(
                    config,
                    power1,
                    power2,
                    lambda_sweep,
                    tanh_factor,
                    tanh_factor2,
                    int_ratio,
                )
                res["R1"]["Learned_KI"] = max_cap_RX1
                res_change["Learned_KI"].append(max_cap_RX1)
                res_pdf_ki = {
                    "RX1_ki": max_pdf_x_RX1,
                    "RX2_ki": max_pdf_x_RX2,
                }
                res_alph_ki = {
                    "RX1_ki": alphabet_x_RX1,
                    "RX2_ki": alphabet_x_RX2,
                }
                res_pdf = {**res_pdf_tin, **res_pdf_ki}
                res_alph = {**res_alph_tin, **res_alph_ki}
                res_opt = {"tin": save_opt_sum_capacity, "ki": save_opt_sum_capacity2}
            else:
                res["R1"]["Learned"] = max_cap_RX1
                res_pdf = {
                    "RX1": max_pdf_x_RX1,
                    "RX2": max_pdf_x_RX2,
                }
                res_alph = {
                    "RX1": alphabet_x_RX1,
                    "RX2": alphabet_x_RX2,
                }
                res_opt = {"opt": save_opt_sum_capacity}

            res["R2"]["Learned"] = max_cap_RX2

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
            )
        plot_R1_R2_curve(
            res, power1, power2, update_save_location, config=config, res_gaus=res_gaus
        )

        io.savemat(
            update_save_location + "res.mat",
            res,
        )

        del alphabet_x_RX1, alphabet_y_RX1, alphabet_x_RX2, alphabet_y_RX2

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
