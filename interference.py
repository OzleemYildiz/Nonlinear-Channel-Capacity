from utils import (
    read_config,
    plot_interference,
    get_interference_alphabet_x_y,
    interference_dependent_snr,
    plot_res,
    plot_R1_R2_curve,
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
    sequential_gradient_descent_on_interference,
    gradient_descent_projection_with_learning_rate,
)


def define_save_location(config):
    save_location = (
        config["output_dir"]
        + "/"
        + config["cons_str"]
        + "_phi="
        + str(config["nonlinearity"])
    )
    if config["nonlinearity"] == 3:
        save_location = save_location + "_tanh=" + str(config["tanh_factor"])

    save_location = (
        save_location
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

    if config["gd_active"]:
        save_location = save_location + "_gd_" + str(config["gd_active"])
        if config["x2_fixed"]:
            save_location = save_location + "_x2_fixed"
            save_location = save_location + "_x2=" + str(config["x2_type"])
            save_location = (
                save_location + "_x1update=" + str(config["x1_update_scheme"])
            )

    save_location = save_location + "/"
    return save_location


def main():
    # First and Third regime implementation
    # ----System Model--- Z channel
    # Y1 = Phi(X1 + AX2 +N11)+ N12
    # Y2 = Phi(X2  + N21)+ N22

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
    for power in np.logspace(
        np.log10(config["min_power_cons"]),
        np.log10(config["max_power_cons"]),
        config["n_snr"],
    ):
        print("-----------Power: ", power, "-----------")
        # snr1, snr2, inr1 = interference_dependent_snr(config, power)
        # cap_RX1_no_int_no_nonlinearity.append(0.5 * np.log2(1 + snr1))
        # cap_RX2_no_nonlinearity.append(0.5 * np.log2(1 + snr2))

        alphabet_x_RX1, alphabet_y_RX1, alphabet_x_RX2, alphabet_y_RX2 = (
            get_interference_alphabet_x_y(config, power)
        )
        res_gaus = {}
        cap1_g, cap2_g = gaussian_interference_capacity(
            power,
            power,
            config,
            alphabet_x_RX1,
            alphabet_y_RX1,
            alphabet_x_RX2,
            alphabet_y_RX2,
        )

        res_gaus["Gaussian"] = [cap1_g, cap2_g]
        # cap1_agc, cap2_agc = agc_gaussian_capacity_interference(config, power)
        # res_gaus["AGC"] = [cap1_agc, cap2_agc]

        cap1, cap2 = gaus_interference_R1_R2_curve(config, power)

        # cap_gaus_RX1.append(cap1)
        # cap_gaus_RX2.append(cap2)

        res = {"R1": {}, "R2": {}}
        res["R1"]["Gaussian"] = cap1
        res["R2"]["Gaussian"] = cap2
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
            ) = gradient_descent_on_interference(config, power, lambda_sweep)

            res["R1"]["Learned"] = max_cap_RX1
            res["R2"]["Learned"] = max_cap_RX2
            res_pdf = {
                "RX1_pdf": max_pdf_x_RX1,
                "RX1_alph": alphabet_x_RX1,
                "RX2_pdf": max_pdf_x_RX2,
                "RX2_alph": alphabet_x_RX2,
            }
            io.savemat(
                save_location + "pdf_pow=" + str(int(power)) + ".mat",
                res_pdf,
            )
            plot_res(
                max_sum_cap,
                save_opt_sum_capacity,
                max_pdf_x_RX1,
                max_pdf_x_RX2,
                alphabet_x_RX1,
                alphabet_x_RX2,
                power,
                save_location + "power=" + str(int(power)) + "/",
                lambda_sweep,
            )
        plot_R1_R2_curve(res, power, save_location, config=config, res_gaus=res_gaus)

        io.savemat(save_location + "res_pow=" + str(int(power)) + ".mat", res)

        del alphabet_x_RX1, alphabet_y_RX1, alphabet_x_RX2, alphabet_y_RX2

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
