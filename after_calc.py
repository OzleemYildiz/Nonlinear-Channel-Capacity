import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy import io
from utils import get_regime_class_interference


def main():
    main_loc = "Res-Int-T/Avg_phi=3_regime=3_N=16_sigma11=1_sigma12=1_sigma21=1_sigma22=1_max_iter=10000_lr=[0.0001]_x2_fixed_x2=1_pw1/"
    tanh_factor = 8
    tanh_factor2 = 4
    a = 1
    pw2 = 40
    folder = (
        main_loc
        + "pw2="
        + str(pw2)
        + "_a="
        + str(a)
        + "_k="
        + str(tanh_factor)
        + "_k2="
        + str(tanh_factor2)
        + "/"
    )
    pw1 = 2.0
    inside_folder = folder + "pw1=" + str(pw1) + "/"

    pdf = io.loadmat(inside_folder + "pdf.mat")
    config = io.loadmat(main_loc + "config.mat")
    breakpoint()
    
    # I should also save pdf_y so I can reach that here
    
    
    regime_RX1, regime_RX2= get_regime_class_interference(
        config=config,
        alphabet_x_RX1=alphabet_x_RX1,
        alphabet_x_RX2=alphabet_x_RX2,
        alphabet_y_RX1=alphabet_y_RX1,
        alphabet_y_RX2=alphabet_y_RX2,
        power1=pw1,
        power2=pw2,
        tanh_factor=tanh_factor,
        tanh_factor2=tanh_factor2,
    )


if __name__ == "__main__":
    main()
