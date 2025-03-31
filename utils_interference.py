import numpy as np
from nonlinearity_utils import (
    Hardware_Nonlinear_and_Noise,
    get_derivative_of_nonlinear_fn,
)
import torch
from utils import (
    loss,
    get_interference_alphabet_x_y,
    get_regime_class_interference,
    get_interference_alphabet_x_y_complex,
)

from gd import get_fixed_interferer
from scipy import io


def define_save_location(config):

    save_location = config["output_dir"] + "/"
    if config["complex"]:
        save_location = save_location + "Complex_"

    save_location = save_location + config["cons_str"]
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
        save_location = save_location + "_phi=" + str(config["nonlinearity"])
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

    if config["hardware_params_active"]:
        save_location = (
            save_location
            + "_"
            + config["hd_change"]
            + "_fixed_SNR="
            + str(config["snr_not_change"])
            + "_"
        )
    else:
        save_location = save_location + "_" + config["change"] + "_"

    # Add sigmas to the title of the config
    if config["hardware_params_active"]:
        hn = Hardware_Nonlinear_and_Noise(config)
        config["E_sat"] = hn.Esat_lin

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
        str_print = (
            "**** AWGN Interference Channel with Nonlinearity by Hardware with Esat: "
            + str(config["E_sat"])
            + "for Regime: "
            + str(config["regime"])
            + " ****"
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
        str_print = (
            "**** AWGN Interference Channel with Nonlinearity: "
            + str(config["nonlinearity"])
            + "for Regime: "
            + str(config["regime"])
            + " ****"
        )

    return save_location, config, str_print


def change_parameters_range(config):
    if config["hardware_params_active"]:

        hn = Hardware_Nonlinear_and_Noise(config)
        config["E_sat"] = hn.Esat_lin
        # Currently, noises of both users are same
        config["sigma_11"], config["sigma_12"] = hn.noise1_std, hn.noise2_std
        config["sigma_21"], config["sigma_22"] = config["sigma_11"], config["sigma_12"]

        config["min_power_cons"], config["max_power_cons"] = (
            hn.P_in_min_linear,
            hn.P_in_max_linear,
        )
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


def get_run_parameters(config, chng, reg3_active=False):

    if config["hardware_params_active"]:
        hn = Hardware_Nonlinear_and_Noise(config)
        power1 = hn.get_power_fixed_from_SNR(config["snr_not_change"])
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
        res_str_run = "_pw1=" + "{:.2e}".format(power1, 3)
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
        res_str_run = "_pw2=" + "{:.2e}".format(power2, 3)
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
     
    
     
    if reg3_active:
        power2 = power2 + config["sigma_11"] ** 2 # Sum the variance
     
     
    return power1, power2, int_ratio, tanh_factor, tanh_factor2, res_str, res_str_run


def get_linear_interference_capacity(power1, power2, int_ratio, config, reg3_active=False):
    # This is X2 fixed results --- X1's tin and ki results
    # This is for USER 1
    if config["hardware_params_active"]:
        hn = Hardware_Nonlinear_and_Noise(config)

    if config["regime"] == 3:
        if config["hardware_params_active"] and not config["gain_later"]:
            sigma_1_snr = config["sigma_11"] ** 2 * hn.gain_lin
        else:
            sigma_1_snr = config["sigma_11"] ** 2
        noise_power = sigma_1_snr + config["sigma_12"] ** 2

    elif config["regime"] == 1:
        noise_power = config["sigma_12"] ** 2
    else:
        raise ValueError("Regime not defined")

    if config["hardware_params_active"] and not config["gain_later"]:

        power1_snr = power1 * hn.gain_lin
        int_snr = int_ratio**2 * power2 * hn.gain_lin

    else:
        power1_snr = power1
        int_snr = int_ratio**2 * power2
        
    if reg3_active: # This assumes gain_later is True
        if config["hardware_params_active"] and not config["gain_later"]:
            sigma_1_snr = config["sigma_11"] ** 2 * hn.gain_lin
        else:
            sigma_1_snr = config["sigma_11"] ** 2
        noise_power = sigma_1_snr + config["sigma_12"] ** 2

        
        int_snr = (power2-config["sigma_11"]**2) * int_ratio**2


    snr_linear_ki = power1_snr / noise_power
    snr_linear_tin = power1_snr / (noise_power + int_snr)

    linear_ki = 1 / 2 * np.log(1 + snr_linear_ki)

    linear_tin = 1 / 2 * np.log(1 + snr_linear_tin)

    if config["complex"]:  # 1/2 comes from real
        linear_ki = linear_ki * 2
        linear_tin = linear_tin * 2

    return linear_tin, linear_ki


def get_interferer_capacity(config, power2):
    if config["hardware_params_active"] and not config["gain_later"]:
        hn = Hardware_Nonlinear_and_Noise(config)
        power2_snr = power2 * hn.gain_lin
    else:
        power2_snr = power2

    if config["regime"] == 3:
        if config["hardware_params_active"] and not config["gain_later"]:
            sigma1_snr = config["sigma_21"] ** 2 * hn.gain_lin
        else:
            sigma1_snr = config["sigma_21"] ** 2
        noise_power = sigma1_snr + config["sigma_22"] ** 2

    elif config["regime"] == 1:
        noise_power = config["sigma_22"] ** 2
    else:
        raise ValueError("Regime not defined")

    cap = 1 / 2 * np.log(1 + power2_snr / (noise_power))
    if config["complex"]:  # 1/2 comes from real
        cap = cap * 2

    return cap


def get_linear_approximation_capacity(regime_RX2, config, power1, pdf_x_RX2):

    regime_RX2.fix_with_multiplying()
    deriv_func = get_derivative_of_nonlinear_fn(
        regime_RX2.config, tanh_factor=regime_RX2.tanh_factor
    )
    power1 = power1 / (10 ** (regime_RX2.multiplying_factor))

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

    ki = torch.log(
        1
        + (d_phi_s * power1)
        / (config["sigma_11"] ** 2 * d_phi_s + config["sigma_12"] ** 2)
    )

    approx_cap_ki = ki @ pdf_x_RX2

    tin = torch.log(
        1
        + (d_phi_s * power1)
        / (
            config["sigma_11"] ** 2 * d_phi_s
            + config["sigma_12"] ** 2**2
            + abs(regime_RX2.get_out_nonlinear(regime_RX2.alphabet_x)) ** 2
        )
    )
    approx_cap_tin = tin @ pdf_x_RX2

    if not config["complex"]:
        approx_cap_ki = approx_cap_ki / 2
        approx_cap_tin = approx_cap_tin / 2
    regime_RX2.unfix_with_multiplying()
    power1 = power1 * (10 ** (regime_RX2.multiplying_factor))

    return approx_cap_ki.detach().numpy(), approx_cap_tin.detach().numpy()


def get_tdm_capacity_with_optimized_dist(
    regime_RX1, regime_RX2, config, pdf_x_RX2, save_location, save_active=False
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
    if save_active:
        io.savemat(save_location + "/tdm.mat", res_tdm)

    return res_tdm


def get_updated_params_for_hd(config, power1, power2):
    if config["regime"] == 1:
        min_noise_pw = config["sigma_12"] ** 2
    elif config["regime"] == 3:
        min_noise_pw = min(config["sigma_11"] ** 2, config["sigma_12"] ** 2)
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
    config["sigma_11"] = config["sigma_11"] * 10 ** (multiplying_factor / 2)
    config["sigma_12"] = config["sigma_12"] * 10 ** (multiplying_factor / 2)
    config["sigma_21"] = config["sigma_21"] * 10 ** (multiplying_factor / 2)
    config["sigma_22"] = config["sigma_22"] * 10 ** (multiplying_factor / 2)
    config["iip3"] = config["iip3"] + 10 * multiplying_factor
    config["multiplying_factor"] = multiplying_factor
    return power1, power2, config


def fix_config_for_hd(config):
    config["sigma_11"] = config["sigma_11"] / 10 ** (config["multiplying_factor"] / 2)
    config["sigma_12"] = config["sigma_12"] / 10 ** (config["multiplying_factor"] / 2)
    config["sigma_21"] = config["sigma_21"] / 10 ** (config["multiplying_factor"] / 2)
    config["sigma_22"] = config["sigma_22"] / 10 ** (config["multiplying_factor"] / 2)
    config["iip3"] = config["iip3"] - 10 * config["multiplying_factor"]

    return config


def get_int_regime(config, power1, power2, int_ratio, tanh_factor, tanh_factor2):
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
            multiplying_factor=config["multiplying_factor"],
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
            multiplying_factor=config["multiplying_factor"],
        )
    return regime_RX1, regime_RX2
