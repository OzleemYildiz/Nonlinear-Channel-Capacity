import numpy as np
from utils import read_config
from nonlinearity_utils import Nonlinearity_Noise, Hardware_Nonlinear_and_Noise
import os
from utils_int import (
    get_save_location,
    get_power,
    get_linear_int_capacity,
    get_capacity_learned,
    get_capacity_gaussian,
    plot_int_res,
    plot_int_pdf,
    get_linear_app_int_capacity,
)
from utils_interference import get_int_regime, get_updated_params_for_hd, fix_config_for_hd
from scipy import io
from gd import get_fixed_interferer


# In this file, we always take the X_2 fixed during interference - with Gaussian

# TODO:
# 1. TDM could be added
# 2. Approximation is not working great- Gotta fix that


def main():
    config = read_config(title_active=False)
    save_location = get_save_location(config)
    os.makedirs(save_location, exist_ok=True)

    int_ratio = config["int_ratio"]

    change_range = np.linspace(
        config["snr_min_dB"],
        config["snr_min_dB"] + config["snr_range"],
        config["n_change"],
    )
    if config["dB_definition_active"]:
        nonlinear_class = Nonlinearity_Noise(config)
    elif config["hardware_params_active"]:
        nonlinear_class = Hardware_Nonlinear_and_Noise(config)

    # Add Noises in the Config
    config = nonlinear_class.update_config(config)

    # Everything is for the First USER - RX1
    res = {"KI": {}, "TIN": {}}

    res["KI"] = {
        "Linear": [],
        "Gaussian": [],
    }
    res["TIN"] = {
        "Linear": [],
        "Gaussian": [],
    }

    if config["gd_active"]:
        res["KI"]["Optimized"] = []
        res["TIN"]["Optimized"] = []
    res["Change_Range"] = change_range

    # res["KI"]["Approximation"] = []

    if config["gd_active"]:
        pdf = {
            "KI": [],
            "TIN": [],
        }
        alph = []

    reg3_active = config["reg3_active"] and config["regime"]==1 and config["x2_fixed"] and config["x2_type"]==0
    
    for ind_c, chn in enumerate(change_range):
        print(
            "************** Change over "
            + str(config["change"])
            + ": "
            + str(chn)
            + " **************"
        )

        power1, power2 = get_power(chn, nonlinear_class, config)
        if config["hardware_params_active"]:
            power1, power2, config = get_updated_params_for_hd(config, power1, power2)
            
        noise_power = nonlinear_class.get_total_noise_power()
        linear_ki, linear_tin = get_linear_int_capacity(
            power1, power2,  int_ratio, config, reg3_active
        )

        regime_RX1, regime_RX2 = get_int_regime(
            config, power1, power2, int_ratio, tanh_factor=0, tanh_factor2=0
        )
        
        save_loc_rx2 = save_location + "/pdf_x_RX2.png"
        pdf_x_RX2 = get_fixed_interferer(
            config, regime_RX2, config["x2_type"], save_loc_rx2
        )
       
        # approx_cap_ki = get_linear_app_int_capacity(
        #     regime_RX2, config, power1, pdf_x_RX2, int_ratio
        # )

        cap_g_ki, cap_g_tin = get_capacity_gaussian(
            regime_RX1, regime_RX2, pdf_x_RX2, int_ratio,reg3_active=reg3_active
        )
        print("Linear KI: ", linear_ki)
        print("Gaussian KI: ", cap_g_ki)
        
        
        if config["gd_active"]:
            cap_learned_ki, cap_learned_tin, pdf_learned_tin, pdf_learned_ki = (
                get_capacity_learned(
                    regime_RX1,
                    regime_RX2,
                    pdf_x_RX2,
                    int_ratio,
                    config,
                    save_location,
                    chn,
                    reg3_active=reg3_active,
                )
            )
            # Save the results
            res["KI"]["Optimized"].append(cap_learned_ki)
            res["TIN"]["Optimized"].append(cap_learned_tin)

            pdf["KI"].append(pdf_learned_ki[0])
            pdf["TIN"].append(pdf_learned_tin[0])
            alph.append(regime_RX1.alphabet_x)

        res["TIN"]["Linear"].append(linear_tin)
        res["KI"]["Linear"].append(linear_ki)
        # res["KI"]["Approximation"].append(approx_cap_ki)
        res["KI"]["Gaussian"].append(cap_g_ki)
        res["TIN"]["Gaussian"].append(cap_g_tin)

        # Free the memory
        del pdf_x_RX2, regime_RX1, regime_RX2
        if config["hardware_params_active"]:
            config = fix_config_for_hd(config)

    # plot_results
    plot_int_res(res, config, save_location, change_range)

    # Save the results

    io.savemat(save_location + "results_tin.mat", res["TIN"])
    io.savemat(save_location + "results_ki.mat", res["KI"])

    io.savemat(save_location + "config.mat", config)

    if config["gd_active"]:
        plot_int_pdf(pdf, config, save_location, change_range, alph)
        alphabet = {"alphabet": alph}
        io.savemat(save_location + "pdf.mat", pdf)
        io.savemat(save_location + "alphabet.mat", alphabet)

    print("Saved in ", save_location)


if __name__ == "__main__":
    main()
