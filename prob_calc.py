import os
from utils import (
    read_config,
    get_alphabet_x_y,
    get_PP_complex_alphabet_x_y,
    get_regime_class,
)
import time
from nonlinearity_utils import return_nonlinear_fn_numpy
import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy.io as io
from gaussian_capacity import get_gaussian_distribution
from PIL import Image
import matplotlib.animation as animation


def monte_carlo(config, alphabet_x, alphabet_y):
    pdf_y_given_x = np.zeros((config["y_len"], len(alphabet_x)))
    y_loc_given_x = np.zeros((config["y_len"], len(alphabet_x)))
    y_bins = np.hstack(
        [
            np.array(alphabet_y) - np.diff(alphabet_y)[0] / 2,
            alphabet_y[-1] + np.diff(alphabet_y)[0] / 2,
        ]
    )

    y = pass_the_channel(alphabet_x, config)
    for ind, x in enumerate(alphabet_x):
        hist, bins = np.histogram(y[ind, :], bins=y_bins)
        pdf_y_given_x[:, ind] = hist / config["sample_num"]
        y_loc_given_x[:, ind] = (bins[:-1] + bins[1:]) / 2
    return pdf_y_given_x, y_loc_given_x


def pass_the_channel(x, config):
    phi = return_nonlinear_fn_numpy(config, tanh_factor=config["tanh_factor1"])
    if config["regime"] == 3:
        n_1 = np.random.normal(0, config["sigma_1"], config["sample_num"])
        n_2 = np.random.normal(0, config["sigma_2"], config["sample_num"])
        y = phi(x.reshape(-1, 1) + n_1.reshape(1, -1)) + n_2.reshape(1, -1)
    elif config["regime"] == 1:
        n = np.random.normal(0, config["sigma_1"], config["sample_num"])
        y = phi(x.reshape(-1, 1)) + n.reshape(1, -1)
    else:
        raise ValueError("Regime not defined")
    return y


def run_comp(config, save_location):

    if not config["interference_active"]:

        # real_x, imag_x, real_y, imag_y = get_PP_complex_alphabet_x_y(
        #     config, config["power1"], config["tanh_factor1"]
        # )
        # regime_class = get_regime_class(
        #     config=config,
        #     alphabet_x=real_x,
        #     alphabet_y=real_y,
        #     power=config["power1"],
        #     tanh_factor=config["tanh_factor1"],
        #     alphabet_x_imag=imag_x,
        #     alphabet_y_imag=imag_y,
        # )
        alphabet_x, alphabet_y, max_x, max_y = get_alphabet_x_y(
            config, config["power1"], config["tanh_factor1"]
        )
        y_len = len(alphabet_y)
        config["y_len"] = y_len
        regime_class = get_regime_class(
            config, alphabet_x, alphabet_y, config["power1"], config["tanh_factor1"]
        )
        res = {}
        st_time = time.time()
        pdf_y_given_x_calculated = regime_class.get_pdf_y_given_x()
        res["analytical_time"] = time.time() - st_time
        print("Time taken for analytical calculation: ", res["analytical_time"])
        # print(torch.sum(pdf_y_given_x_calculated, axis=0))

        st_time
        mc_pdf_y_given_x, mc_y_loc_given_x = monte_carlo(config, alphabet_x, alphabet_y)
        res["mc_time"] = time.time() - st_time
        print("Time taken for MC calculation: ", res["mc_time"])
        # print(np.sum(mc_pdf_y_given_x, axis=0))

        pdf_x = get_gaussian_distribution(
            config["power1"], regime_class, config["complex"]
        )
        mut_info_analytical = regime_class.new_capacity(
            pdf_x=pdf_x, pdf_y_given_x=pdf_y_given_x_calculated
        )
        mut_info_mc = regime_class.new_capacity(
            pdf_x=pdf_x, pdf_y_given_x=torch.tensor(mc_pdf_y_given_x).float()
        )
        res["mut_info_analytical"] = mut_info_analytical
        res["mut_info_mc"] = mut_info_mc
        res["Error"] = np.abs(mut_info_analytical - mut_info_mc)
        print(
            "Error in Mutual Information: ",
            res["Error"],
            "when the real one is ",
            mut_info_analytical,
            " MC one is",
            mut_info_mc,
        )
        images = []

        fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)
        ind = 0
        line1 = ax.plot(
            mc_y_loc_given_x[:, ind],
            mc_pdf_y_given_x[:, ind],
            label="MC",
        )
        line2 = ax.plot(
            alphabet_y,
            pdf_y_given_x_calculated[:, ind],
            label="Analytical",
        )
        ax.set_xlabel("y", fontsize=10)
        ax.set_ylabel("pdf(y|x)", fontsize=10)
        ax.legend(loc="best", fontsize=10)
        ax.grid(
            visible=True,
            which="major",
            axis="both",
            color="lightgray",
            linestyle="-",
            linewidth=0.5,
        )
        plt.minorticks_on()
        ax.grid(
            visible=True,
            which="minor",
            axis="both",
            color="gainsboro",
            linestyle=":",
            linewidth=0.5,
        )

        def update(ind):
            line1[0].set_ydata(mc_pdf_y_given_x[:, ind])
            line1[0].set_xdata(mc_y_loc_given_x[:, ind])
            line2[0].set_ydata(pdf_y_given_x_calculated[:, ind])
            ax.set_title("x=" + str(alphabet_x[ind]))
            return line1, line2

        ani = animation.FuncAnimation(
            fig=fig, func=update, frames=range(len(alphabet_x)), interval=300
        )

        ani.save(filename=save_location + "/Comp.gif", writer="pillow")

        io.savemat(save_location + "res.mat", res)


def plot_res(rand_n, min_sample, main_save_location):

    error = []
    time_mc = []
    time_analytical = []

    for r in rand_n:
        name_file = main_save_location + "/N_Random=" + str(int(r)) + "/res.mat"
        res = io.loadmat(name_file)
        error.append(res["Error"][0, 0])
        time_mc.append(res["mc_time"][0, 0])
        time_analytical.append(res["analytical_time"][0, 0])
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), tight_layout=True)

    ax1.plot(rand_n, error, label="Error")
    ax1.set_xlabel("# of Random", fontsize=10)
    ax1.set_ylabel("Error in Mutual Information Calculation", fontsize=10)
    ax1.legend(loc="best", fontsize=10)
    ax1.set_title("Min Sample: " + str(min_sample))
    ax1.grid(
        visible=True,
        which="major",
        axis="both",
        color="lightgray",
        linestyle="-",
        linewidth=0.5,
    )
    plt.minorticks_on()
    ax1.grid(
        visible=True,
        which="minor",
        axis="both",
        color="gainsboro",
        linestyle=":",
        linewidth=0.5,
    )
    ax2.plot(rand_n, time_mc, label="MC Time")
    ax2.plot(rand_n, time_analytical, label="Analytical Time")
    ax2.set_xlabel("Number of Random Number Generations", fontsize=10)
    ax2.set_ylabel("Time (s)", fontsize=10)
    ax2.legend(loc="best", fontsize=10)
    ax2.set_title("Min Sample: " + str(min_sample))
    ax2.grid(
        visible=True,
        which="major",
        axis="both",
        color="lightgray",
        linestyle="-",
        linewidth=0.5,
    )
    plt.minorticks_on()
    ax2.grid(
        visible=True,
        which="minor",
        axis="both",
        color="gainsboro",
        linestyle=":",
        linewidth=0.5,
    )

    fig.savefig(
        main_save_location + "Comp.png",
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":

    config = read_config(args_name="args_prob.yml")

    rand_n = np.linspace(100, 1000, 2)
    main_save_location = (
        "Prob_Compare/" + "MinSample=" + str(config["min_samples"]) + "/"
    )
    os.makedirs(main_save_location, exist_ok=True)

    for r in rand_n:
        config["sample_num"] = int(r)
        save_location = (
            main_save_location + "/N_Random=" + str(config["sample_num"]) + "/"
        )
        os.makedirs(save_location, exist_ok=True)
        run_comp(config, save_location)

    plot_res(rand_n, config["min_samples"], main_save_location)
