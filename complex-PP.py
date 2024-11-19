import yaml
from utils import read_config, get_PP_complex_alphabet_x_y, regime_dependent_snr
import os
import numpy as np


def define_save_location(config):
    save_location = (
        config["output_dir"]
        + "/"
        + config["cons_str"]
        + "_phi="
        + str(config["nonlinearity"])
        + "_regime="
        + str(config["regime"])
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

    for snr in snr_change:
        print("SNR Change: ", snr)
        power = (10 ** (snr / 10)) * noise_power

        cap_no_nonlinear.append(np.log(1 + power / (noise_power)))
        real_x, imag_x, real_y, imag_y = get_PP_complex_alphabet_x_y(config, power)


if __name__ == "__main__":
    main()
