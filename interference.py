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


def define_save_location(config):

    save_location = config["output_dir"] + "/"
    if config["complex"]:
        save_location = save_location + "Complex_"

    save_location = (
        save_location + config["cons_str"] + "_phi=" + str(config["nonlinearity"])
    )
    if config["hardware_params_active"]:
        save_location = (
            save_location
            + "_hw_gain="
            + str(config["gain"])
            + "_iip3="
            + str(config["iip3"])
            + "_bw="
            + str(config["bandwidth"])
        )
        if config["regime"] == 3:
            save_location = save_location + "_NF1=" + str(config["noise_figure1"])
        save_location = save_location + "_NF2=" + str(config["noise_figure2"])

    else:
        if config["nonlinearity"] == 5:
            save_location = save_location + "_clip=" + str(config["clipping_limit_x"])
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

    if config["ADC"]:
        save_location = save_location + "_ADC_b=" + str(config["bits"])

    save_location = (
        save_location
        + "_regime="
        + str(config["regime"])
        + "_N="
        + str(config["min_samples"])
    )

    if config["gd_active"]:
        save_location = (
            save_location
            + "_max_iter="
            + str(config["max_iter"])
            + "_lr="
            + str(config["lr"])
        )
        if config["x2_fixed"]:

            save_location = save_location + "_x2_fixed"
            save_location = save_location + "_x2=" + str(config["x2_type"])

    save_location = save_location + "_" + config["change"] + "/"

    return save_location


def change_parameters_range(config):
    if config["hardware_params_active"]:
        hn = Hardware_Nonlinear_and_Noise(config)
        config["E_sat"] = hn.Esat_lin
        config["sigma_11"], config["sigma_12"] = hn.get_noise_vars()
        config["sigma_21"], config["sigma_22"] = config["sigma_11"], config["sigma_12"]
        config["min_power_cons"], config["max_power_cons"] = hn.get_min_max_power()
        config["change"] = config["hd_change"]

    if config["change"] == "pw1":
        if config["hardware_params_active"]:
            config["min_power1"], config["max_power1"] = (
                config["min_power_cons"],
                config["max_power_cons"],
            )
        change_range = np.logspace(
            np.log10(config["min_power1"]),
            np.log10(config["max_power1"]),
            config["n_change"],
        )
        print("-------------Change is Power1-------------")
    elif config["change"] == "pw2":
        if config["hardware_params_active"]:
            config["min_power2"], config["max_power2"] = (
                config["min_power_cons"],
                config["max_power_cons"],
            )

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
    elif config["change"] == "k" and not config["hardware_params_active"]:
        if config["nonlinearity"] != 3:
            raise ValueError("Tanh Factor is only for nonlinearity 3")
        change_range = np.linspace(
            config["min_tanh_factor"],
            config["max_tanh_factor"],
            config["n_change"],
        )
        print("-------------Change is Tanh Factor for User 1-------------")
    elif config["change"] == "k2" and not config["hardware_params_active"]:
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

    return change_range, config


def get_run_parameters(config, chng):

    if config["hardware_params_active"]:
        hn = Hardware_Nonlinear_and_Noise(config)
        power1 = hn.get_power_fixed(config["snr_not_change"])
        power2 = power1
        int_ratio = config["min_int_ratio"]
        tanh_factor = np.sqrt(hn.Esat_lin)
        tanh_factor2 = tanh_factor
    else:
        power1 = config["min_power1"]
        power2 = config["min_power2"]
        int_ratio = config["min_int_ratio"]
        tanh_factor = config["min_tanh_factor"]
        tanh_factor2 = config["min_tanh_factor_2"]

    if config["change"] == "pw1":
        print("Power1: ", chng)
        power1 = chng
        res_str = (
            "pw2="
            + str(power2)
            + "_a="
            + str(int_ratio)
            + "_k="
            + str(tanh_factor)
            + "_k2="
            + str(tanh_factor2)
        )
        res_str_run = "_pw1=" + str(round(power1, 3))
    elif config["change"] == "pw2":
        print("Power2: ", chng)
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
        res_str_run = "_pw2=" + str(round(power2, 3))
    elif config["change"] == "a":
        print("Int Ratio: ", chng)
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
        res_str_run = "_a=" + str(int_ratio)
    elif config["change"] == "k" and not config["hardware_params_active"]:
        print("Tanh Factor: ", chng)
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
        res_str_run = "_k=" + str(tanh_factor)
    elif config["change"] == "k2" and not config["hardware_params_active"]:
        print("Tanh Factor2: ", chng)
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
        res_str_run = "_k2=" + str(tanh_factor2)

    return power1, power2, int_ratio, tanh_factor, tanh_factor2, res_str, res_str_run


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

    if config["complex"]:  # 1/2 comes from real
        linear_ki = linear_ki * 2
        linear_tin = linear_tin * 2

    return linear_tin, linear_ki


def get_interferer_capacity(config, power2):
    if config["regime"] == 3:
        cap = (
            1
            / 2
            * np.log(1 + power2 / (config["sigma_21"] ** 2 + config["sigma_22"] ** 2))
        )
    elif config["regime"] == 1:
        cap = 1 / 2 * np.log(1 + power2 / config["sigma_22"] ** 2)
    else:
        raise ValueError("Regime not defined")

    if config["complex"]:  # 1/2 comes from real
        cap = cap * 2

    return cap


def get_linear_approximation_capacity(regime_RX2, config, power1, pdf_x_RX2):

    deriv_func = get_derivative_of_nonlinear_fn(
        config, tanh_factor=regime_RX2.tanh_factor
    )
    if config["complex"]:
        d_phi_s = (
            abs(
                deriv_func(abs(regime_RX2.alphabet_x))
                * torch.exp(1j * torch.angle(regime_RX2.alphabet_x))
            )
            ** 2
        )
    else:
        d_phi_s = abs(deriv_func(regime_RX2.alphabet_x)) ** 2
    approx_cap1 = torch.mean(
        torch.log(
            1
            + (d_phi_s * power1)
            / (config["sigma_11"] ** 2 * d_phi_s + config["sigma_12"] ** 2)
        )
        * pdf_x_RX2
    )
    if not config["complex"]:
        approx_cap1 = approx_cap1 / 2

    return approx_cap1.detach().numpy()


def get_tdm_capacity_with_optimized_dist(
    regime_RX1, regime_RX2, config, pdf_x_RX2, save_location
):
    print("---- TDM Capacity ----")

    if (
        config["x2_type"] == 0 or not config["x2_fixed"]
    ):  # calculated pdf_x_RX2 is gaussian and I need an optimized distribution
        save_rx2 = save_location + "/TDM_pdf_x_RX2_opt.png"
        pdf_x_RX2 = get_fixed_interferer(
            config, regime_RX2, 1, save_rx2
        )  # optimized X2

    cap_RX2 = loss(pdf_x_RX2, regime_RX2)
    cap_RX2 = -cap_RX2

    save_rx1 = save_location + "/TDM_pdf_x_RX1_opt.png"

    pdf_x_RX1 = get_fixed_interferer(config, regime_RX1, 1, save_rx1)  # optimized X1
    cap_RX1 = loss(pdf_x_RX1, regime_RX1)
    cap_RX1 = -cap_RX1

    time_lambda = np.linspace(0, 1, 100)

    res_tdm = {}
    res_tdm["R1"] = cap_RX1.detach().numpy() * time_lambda
    res_tdm["R2"] = cap_RX2.detach().numpy() * (1 - time_lambda)
    return res_tdm


def main():
    # First and Third regime implementation
    # ----System Model--- Z channel
    # Y1 = Phi(X1 + AX2 +N11)+ N12
    # Y2 = Phi(X2  + N21)+ N22

    st = time.time()
    config = read_config()
    if config["hardware_params_active"]:
        hn = Hardware_Nonlinear_and_Noise(config)
        config["E_sat"] = hn.Esat_lin

    # Add sigmas to the title of the config
    if config["hardware_params_active"]:
        config["title"] = (
            config["title"]
            + " Hardware Parameters, Esat: "
            + str(config["E_sat"])
            + " IIP3: "
            + str(config["iip3"])
            + " BW: "
            + str(config["bandwidth"])
            + " NF1: "
            + str(config["noise_figure1"])
            + " NF2: "
            + str(config["noise_figure2"])
            + " Gain: "
            + str(config["gain"])
        )
    else:
        if config["regime"] == 1:  # Y = phi(X1) + N2
            config["title"] = (
                config["title"]
                + "_sigma12="
                + str(config["sigma_12"])
                + "_sigma22="
                + str(config["sigma_22"])
            )
        elif config["regime"] == 3:  # Y = phi(X1 + N1) + N2
            config["title"] = (
                config["title"]
                + "_sigma11="
                + str(config["sigma_11"])
                + "_sigma12="
                + str(config["sigma_12"])
                + "_sigma21="
                + str(config["sigma_21"])
                + "_sigma22="
                + str(config["sigma_22"])
            )

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

    change_range, config = change_parameters_range(config)
    res_change = {
        "Linear_TIN": [],
        "Gaussian_TIN": [],
        "Linear_KI": [],
        "Gaussian_KI": [],
    }
    if config["regime"] == 3:
        res_change["Linear_Approx"] = []

    if config["gd_active"]:
        res_change["Learned_KI"] = []
        res_change["Learned_TIN"] = []
    old_config_title = config["title"]

    _, _, _, _, _, res_str, _ = get_run_parameters(config, change_range[0])

    save_location = save_location + res_str + "/"
    os.makedirs(save_location, exist_ok=True)
    for ind, chng in enumerate(change_range):
        power1, power2, int_ratio, tanh_factor, tanh_factor2, res_str, res_str_run = (
            get_run_parameters(config, chng)
        )

        if config["hardware_params_active"]:
            sigma_1, sigma_2 = hn.get_noise_vars()
            if config["regime"] == 1:
                min_noise_pw = sigma_2**2
            elif config["regime"] == 3:
                min_noise_pw = min(sigma_1**2, sigma_2**2)
            multiplying_factor = -round(
                np.log10(
                    min(
                        power1,
                        power2,
                        min_noise_pw,
                    )
                )
            )
            power1 = power1 * 10**multiplying_factor
            power2 = power2 * 10**multiplying_factor
            config["sigma_12"] = config["sigma_12"] * 10 ** (multiplying_factor / 2)
            config["sigma_21"] = config["sigma_21"] * 10 ** (multiplying_factor / 2)
            config["sigma_22"] = config["sigma_22"] * 10 ** (multiplying_factor / 2)
            config["iip3"] = config["iip3"] + 10 * multiplying_factor
        else:
            multiplying_factor = 1
        config["multiplying_factor"] = multiplying_factor

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

        res_change["Linear_TIN"].append(linear_tin)
        res_change["Linear_KI"].append(linear_ki)

        update_save_location = save_location + config["change"] + "=" + str(chng) + "/"
        config["save_location"] = update_save_location

        os.makedirs(update_save_location, exist_ok=True)
        print("----------", str(config["change"]), ":", chng, "-----------")
        if config["complex"]:
            real_x1, imag_x1, real_y1, imag_y1, real_x2, imag_x2, real_y2, imag_y2 = (
                get_interference_alphabet_x_y_complex(
                    config, power1, power2, int_ratio, tanh_factor, tanh_factor2
                )
            )

            regime_RX1, regime_RX2 = get_regime_class_interference(
                config=config,
                alphabet_x_RX1=real_x1,
                alphabet_x_RX2=real_x2,
                alphabet_y_RX1=real_y1,
                alphabet_y_RX2=real_y2,
                power1=power1,
                power2=power2,
                tanh_factor=tanh_factor,
                tanh_factor2=tanh_factor2,
                alphabet_x_RX1_imag=imag_x1,
                alphabet_x_RX2_imag=imag_x2,
                alphabet_y_RX1_imag=imag_y1,
                alphabet_y_RX2_imag=imag_y2,
            )

        else:
            alphabet_x_RX1, alphabet_y_RX1, alphabet_x_RX2, alphabet_y_RX2 = (
                get_interference_alphabet_x_y(
                    config, power1, power2, int_ratio, tanh_factor, tanh_factor2
                )
            )
            regime_RX1, regime_RX2 = get_regime_class_interference(
                config=config,
                alphabet_x_RX1=alphabet_x_RX1,
                alphabet_x_RX2=alphabet_x_RX2,
                alphabet_y_RX1=alphabet_y_RX1,
                alphabet_y_RX2=alphabet_y_RX2,
                power1=power1,
                power2=power2,
                tanh_factor=tanh_factor,
                tanh_factor2=tanh_factor2,
                multiplying_factor=multiplying_factor,
            )
        if config["x2_fixed"]:
            save_loc_rx2 = update_save_location + "/pdf_x_RX2_opt.png"
            pdf_x_RX2 = get_fixed_interferer(
                config, regime_RX2, config["x2_type"], save_loc_rx2
            )
        else:
            pdf_x_RX2 = None

        res_tdm = get_tdm_capacity_with_optimized_dist(
            regime_RX1, regime_RX2, config, pdf_x_RX2, update_save_location
        )

        # Approximated Capacity
        if config["regime"] == 3:
            if config["x2_fixed"]:
                approx_cap1 = get_linear_approximation_capacity(
                    regime_RX2, config, power1, pdf_x_RX2
                )
            else:
                gaus_pdf_x_RX2 = get_fixed_interferer(
                    config,
                    regime_RX2,
                    0,
                )  # give gaussian - not fixed
                approx_cap1 = get_linear_approximation_capacity(
                    regime_RX2, config, power1, gaus_pdf_x_RX2
                )
            res_change["Linear_Approx"].append(approx_cap1)

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
                    "RX1_tin": regime_RX1.alphabet_x,
                    "RX2_tin": regime_RX2.alphabet_x,
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
                    "RX1_ki": regime_RX1.alphabet_x,
                    "RX2_ki": regime_RX2.alphabet_x,
                }
                res_pdf = {**res_pdf_tin, **res_pdf_ki}
                res_alph = {**res_alph_tin, **res_alph_ki}
                res_opt = {"tin": save_opt_sum_capacity, "ki": save_opt_sum_capacity2}
            else:
                res["R1"]["Learned"] = max_cap_RX1
                res["R2"]["Learned"] = max_cap_RX2
                res_pdf = {
                    "RX1": max_pdf_x_RX1,
                    "RX2": max_pdf_x_RX2,
                }
                res_alph = {
                    "RX1": regime_RX1.alphabet_x,
                    "RX2": regime_RX2.alphabet_x,
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
            update_save_location + "res.mat",
            res,
        )
        if config["complex"]:
            del (
                real_x1,
                imag_x1,
                real_y1,
                imag_y1,
                real_x2,
                imag_x2,
                real_y2,
                imag_y2,
                regime_RX1,
                regime_RX2,
            )
        else:
            del (
                alphabet_x_RX1,
                alphabet_y_RX1,
                alphabet_x_RX2,
                alphabet_y_RX2,
                regime_RX1,
                regime_RX2,
            )
        if config["hardware_params_active"]:
            config["sigma_12"] = config["sigma_12"] / 10 ** (multiplying_factor / 2)
            config["sigma_21"] = config["sigma_21"] / 10 ** (multiplying_factor / 2)
            config["sigma_22"] = config["sigma_22"] / 10 ** (multiplying_factor / 2)
            config["iip3"] = config["iip3"] - 10 * multiplying_factor

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
