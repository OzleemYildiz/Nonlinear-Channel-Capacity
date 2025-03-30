import numpy as np
from utils import read_config
from utils_pp import (
    get_regime_pp,
    get_linear_cap,
    get_gaussian_cap,
    get_power_pp,
    get_mmse_cap,
    plot_res_pp,
    plot_pdf_pp,
    get_capacity_learned,
)
from utils_int import get_int_regime
from nonlinearity_utils import Nonlinearity_Noise
import os
from utils_int import (
    get_save_location,
)
from utils_interference import get_int_regime
from scipy import io
from gd import get_fixed_interferer


# In this file, we always take the X_2 fixed during interference - with Gaussian

# TODO:
# 1. TDM could be added
# 2. Approximation is not working great- Gotta fix that


def main():
    config = read_config(title_active=False)
    save_location = get_save_location(config, pp=True)
    os.makedirs(save_location, exist_ok=True)

    change_range = np.linspace(
        config["snr_min"],
        config["snr_min"] + config["snr_range"],
        config["n_change"],
    )
    nonlinear_class = Nonlinearity_Noise(config)

    # Add Noises in the Config
    config = nonlinear_class.update_config(config, pp=True)

    res = {
        "Linear": [],
        "Gaussian": [],
        "MMSE": [],
    }

    if config["gd_active"]:
        res["Optimized"] = []

    pdf = []
    alph = []

    for ind_c, chn in enumerate(change_range):
        print(
            "************** Change over "
            + str(config["change"])
            + ": "
            + str(chn)
            + " **************"
        )

        power1 = get_power_pp(nonlinear_class, chn)

        linear_cap = get_linear_cap(config, power1)

        regime_class = get_regime_pp(config, power1)

        cap_g = get_gaussian_cap(regime_class, power1, config)

        res = get_mmse_cap(regime_class, res)  # added in the function

        if config["gd_active"]:
            cap_learned, pdf_learned = get_capacity_learned(
                regime_class, power1, config, save_location, chn
            )
            # Save the results
            breakpoint()
            res["Optimized"].append(cap_learned)

            alph.append(regime_RX1.alphabet_x)

        res["Linear"].append(linear_cap)
        res["Gaussian"].append(cap_g)

        # Free the memory
        del regime_class

    # plot_results
    plot_res_pp(res, change_range, save_location, config)
    if config["gd_active"]:
        plot_pdf_pp(pdf, alph, save_location, config, change_range)

    # Save the results

    res["Change_Range"] = change_range
    io.savemat(save_location + "results.mat", res)
    io.savemat(save_location + "config.mat", config)

    if config["gd_active"]:
        pdf = {"pdf": pdf}
        io.savemat(save_location + "pdf.mat", pdf)
        alphabet = {"alphabet": alph}
        io.savemat(save_location + "alphabet.mat", alphabet)

    print("Saved in ", save_location)


if __name__ == "__main__":
    main()
