import yaml
from utils import (
    read_config,
    get_PP_complex_alphabet_x_y,
    regime_dependent_snr,
    return_regime_class,
    plot_vs_change,
)
from bounds import (
    bound_backtracing_check,
)
import os
import numpy as np
from gaussian_capacity import complex_gaussian_capacity_PP


def define_save_location(config):
    save_location = (
        config["output_dir"]
        + "/"
        + config["cons_str"]
        + "_phi="
        + str(config["nonlinearity"])
        + "_regime="
        + str(config["regime"])
        + "_qam="
        + str(config["qam_k"])
        + "_min_pow_1="
        + str(config["min_power_cons"])
    )
    os.makedirs(save_location, exist_ok=True)
    return save_location


def main():
    config = read_config(args_name="arguments-complex.yml")
    print(
        "Complex Domain Point to Point Communication for Regime: ",
        config["regime"],
        "with Nonlinearity: ",
        config["nonlinearity"],
    )
    save_location = define_save_location(config)

    print("Save in:", save_location)
    snr_change, noise_power = regime_dependent_snr(config)

    cap_no_nonlinear = []
    cap_gaus = []

    for snr in snr_change:
        print("SNR Change: ", snr)
        power = (10 ** (snr / 10)) * noise_power

        tanh_factor = config[
            "tanh_factor"
        ]  # There could be a loop around this after #FIXME

        cap_no_nonlinear.append(np.log(1 + power / (noise_power)))
        real_x, imag_x, real_y, imag_y = get_PP_complex_alphabet_x_y(
            config, power, tanh_factor
        )
        regime_class = return_regime_class(
            config=config,
            alphabet_x=real_x,
            alphabet_y=real_y,
            power=power,
            tanh_factor=tanh_factor,
            alphabet_x_imag=imag_x,
            alphabet_y_imag=imag_y,
        )

        new_gaus = complex_gaussian_capacity_PP(power, config, regime_class)
        cap_gaus = bound_backtracing_check(cap_gaus, new_gaus)
        print("Capacity without Nonlinearity: ", cap_no_nonlinear[-1])
        print("Gaussian Capacity: ", new_gaus)

    res = {
        "Gaussian_Capacity": cap_gaus,
        "Capacity_without_Nonlinearity": cap_no_nonlinear,
    }

    plot_vs_change(snr_change, res, config, save_location=save_location)
    print("Results saved in: ", save_location)


if __name__ == "__main__":
    main()
