import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import io
from utils import grid_minor
import torch

def plot_R1_R2_curve_for_different_power(main_folder, sub_folder, iip3):
    R1 = {}
    R2 = {}
    for dirpath, dirnames, _ in os.walk(sub_folder):
        for dirname in dirnames:
            res = io.loadmat(sub_folder + "/" + dirname + "/res.mat")
            key = "{:.1e}".format(float(dirname[4:]))
            R1[key] = res["R1"]["Learned"][0][0][0]
            R2[key] = res["R2"]["Learned"][0][0][0]

    fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)
    for keys in R1.keys():
        plt.plot(R1[keys], R2[keys], label="$P_1=$" + str(keys))
    ax.legend(loc="best")
    ax.set_xlabel("Rate User 1")
    ax.set_ylabel("Rate User 2")
    ax.set_title("Regime 3 with IIP3: ")
    ax = grid_minor(ax)
    os.makedirs(main_folder + "Plots", exist_ok=True)
    fig.savefig(main_folder + "Plots/R1_R2_curve_iip3=" + str(iip3) + ".png")
    plt.close()
    print("Plot saved in ", main_folder + "Plots")


def plot_low_power_rate_curve(main_folder, range_power):

    fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)
    for dirpath, dirnames, _ in os.walk(main_folder):
        for dirname in dirnames:
            if dirname != "Plots":
                for sub_dirpath, sub_dirnames, _ in os.walk(main_folder + dirname):
                    for sub_dirname in sub_dirnames:

                        if float(sub_dirname[4:]) / range_power < 10:
                            res = io.loadmat(
                                main_folder + dirname + "/" + sub_dirname + "/res.mat"
                            )

                            ind_s = dirname.find("iip3") + 5
                            ind_e = dirname.find("_bw")

                            key = dirname[ind_s:ind_e]
                            plt.plot(
                                res["R1"]["Learned"][0][0][0],
                                res["R2"]["Learned"][0][0][0],
                                label="IIP3 = " + key,
                            )
    ax.legend(loc="best")
    ax.set_xlabel("Rate User 1")
    ax.set_ylabel("Rate User 2")
    ax.set_title("Regime 3 with Different Hardware Params ")
    ax = grid_minor(ax)
    os.makedirs(main_folder + "Plots", exist_ok=True)
    fig.savefig(main_folder + "Plots/R1_R2_curve_comp.png")
    plt.close()
    print("Plot saved in ", main_folder + "Plots")


def plot_change_over_pw_FixedX2():
    UX = 2
    main_folder = "Paper_Figure/Regime 3 Hardware/Int/New/PW2-Change-X2fixed/"
    fold_name = "/Avg_hw_gain=15.83_iip3=-6.3_bw=500000000_NF1=4.53_NF2=15"
    fold_name2 = "_regime=3_N=100_max_iter=10000_lr=[0.0001]_x2_fixed_x2="
    fixed = "_pw2_fixed_SNR="
    last_part1 = "_pw1=3.5727283815192956e-17"
    last_part2 = "_pw1=1.1297959146728e-15"
    last_part = "_a=1_k=1.082642326745063e-05_k2=1.082642326745063e-05"
    SNR_list = [5, 20]
    x2_type = [0, 1]
    ADC_active = [True, False]
    b = 3
    file_name = "res_change-pw2"
    file_name_end = "_a=1_k=1.0826423267450607e-05_k2=1.0826423267450607e-05.mat"

    markers = ["o", "s", "D", "v", "^", "<", ">", "p", "P", "*", "X", "d", "H"]
    colors = ["b", "g", "r", "c", "m", "y", "k", "w"]
    linestyle = ["-", "--", "-.", ":"]

    for SNR in SNR_list:
        done = False
        fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)
        if SNR == 5:
            last_part_x = last_part1 + last_part
            file_name_x = file_name + last_part1 + file_name_end
        else:
            last_part_x = last_part2 + last_part
            file_name_x = file_name + last_part2 + file_name_end

        for x2 in x2_type:
            for adc in ADC_active:
                subfolder = "SNR=" + str(SNR) + fold_name
                if adc:
                    subfolder = subfolder + "_ADC_b=" + str(b)
                    label = "b=" + str(b)

                    markeredgecolor = "#580F41"
                else:
                    label = ""

                    markeredgecolor = "#FF796C"

                if x2 == 0:
                    label += " RX" + str(UX) + "= G"
                    markersize = 5
                    fillstyle = "full"
                else:
                    label += " RX" + str(UX) + "= GD"
                    markersize = 15
                    fillstyle = "none"
                subfolder += fold_name2 + str(x2) + fixed + str(SNR) + last_part_x

                res_change = io.loadmat(main_folder + subfolder + "/" + file_name_x)

                if not done:
                    ax.plot(
                        res_change["change_range"].reshape(-1),
                        res_change["Linear_KI"].reshape(-1),
                        label="L KI",
                        linewidth=2,
                        marker=markers[4],
                    )
                    ax.plot(
                        res_change["change_range"].reshape(-1),
                        res_change["Linear_TIN"].reshape(-1),
                        label="L TIN",
                        linewidth=2,
                        marker=markers[0],
                    )
                    done = True
                if adc:
                    ax.plot(
                        res_change["change_range"].reshape(-1),
                        res_change["Gaussian_TIN"].reshape(-1),
                        label="G TIN " + label,
                        linewidth=2,
                        marker=markers[0],
                        markersize=markersize,
                        markeredgecolor=markeredgecolor,
                        fillstyle=fillstyle,
                    )
                    ax.plot(
                        res_change["change_range"].reshape(-1),
                        res_change["Gaussian_KI"].reshape(-1),
                        label="G KI " + label,
                        linewidth=2,
                        marker=markers[4],
                        markersize=markersize,
                        markeredgecolor=markeredgecolor,
                        fillstyle=fillstyle,
                    )

                ax.plot(
                    res_change["change_range"].reshape(-1),
                    res_change["Learned_KI"].reshape(-1),
                    label="GD KI " + label,
                    linewidth=2,
                    linestyle=linestyle[2],
                    marker=markers[4],
                    markersize=markersize,
                    markeredgecolor=markeredgecolor,
                    fillstyle=fillstyle,
                )
                ax.plot(
                    res_change["change_range"].reshape(-1),
                    res_change["Learned_TIN"].reshape(-1),
                    label="GD TIN " + label,
                    linewidth=2,
                    linestyle=linestyle[1],
                    marker=markers[0],
                    markersize=markersize,
                    markeredgecolor=markeredgecolor,
                    fillstyle=fillstyle,
                )
        ax.set_title("SNR = " + str(SNR) + " dB for User " + str(UX))
        ax.set_xlabel("Change in $P_1$")
        ax.set_ylabel("Rate User " + str(UX))
        ax.legend(loc="best", fontsize=8)
        ax = grid_minor(ax)
        os.makedirs(main_folder + "Plots", exist_ok=True)
        fig.savefig(
            main_folder + "Plots/R_change_SNR=" + str(SNR) + ".png",
            bbox_inches="tight",
        )
        plt.close()



def plot_bits_tin_ki_x2_fixed():
    folder = "Paper_Figure/Regime1HD/"
    subfolder = "Avg_hw_gain=15.83_iip3=-6.3_bw=500000000_NF2=15_ADC_b="
    subfolder_2 = "_regime=1_N=100_max_iter=50000_lr=[0.0001]_x2_fixed_x2=0_pw1_fixed_SNR=50_pw2=1.1297959146727999e-12_a=1_k=1.082642326745063e-05_k2=1.082642326745063e-05"

    name = "/res_change-pw1_pw2=1.1297959146727999e-12_a=1_k=1.0826423267450607e-05_k2=1.0826423267450607e-05.mat"
    markers = ["o", "s", "D", "v", "^", "<", ">", "p", "P", "*", "X", "d", "H"]
    linestyes = ["-", "--", "-.", ":"]
    colors = ["b", "g", "r", "c", "m", "y", "k", "w"]
    ind =0
    fig1, ax1 = plt.subplots(figsize=(5, 4), tight_layout=True)
    fig2, ax2 = plt.subplots(figsize=(5, 4), tight_layout=True)
    
    fig3, ax3 = plt.subplots(figsize=(5, 4), tight_layout=True)
    fig4, ax4 = plt.subplots(figsize=(5, 4), tight_layout=True)
    high_power_folder = "/pw1=1.1297959146727999e-14"
    for i in range(3,7):
        res = io.loadmat(folder + subfolder + str(i) + subfolder_2 + name)
        config = io.loadmat(folder + subfolder + str(i) + subfolder_2 + "/config.mat")
        pw2=1.1297959146727999e-12
        inr = 10*np.log10(pw2/config["sigma_11"]**2)[0][0]
        pw1=1.1297959146727999e-14
        fixed_snr = 10*np.log10(pw1/config["sigma_11"]**2)[0][0]
        snr= 10*np.log10(res["change_range"]/config["sigma_11"]**2)
        ax1.plot(
            snr.reshape(-1),
            res["Linear_KI"].reshape(-1),
            label="Linear, b=" + str(i),
            linewidth=2,
            marker = markers[0],
            linestyle = linestyes[0],
            color= colors[ind],
        )
        ax1.plot(
            snr.reshape(-1),
            res["Learned_KI"].reshape(-1),
            label="GD, b=" + str(i),
            linewidth=2,
            marker = markers[1],
            linestyle = linestyes[1],
            color= colors[ind],
        )
        ax1.plot(
            snr.reshape(-1),
            res["Gaussian_KI"].reshape(-1),
            label="Gaussian, b=" + str(i),
            linewidth=2,
            marker = markers[2],
            linestyle = linestyes[2],
            color= colors[ind],
        )
        ax1.plot(snr.reshape(-1), 
                 res["Linear_Approx_KI"].reshape(-1), 
                 label="Approximation, b=" + str(i), 
                 linewidth=2,
                 marker = markers[3],
                linestyle = linestyes[3],color= colors[ind],)
        
        
        snr= 10*np.log10(res["change_range"]/config["sigma_11"]**2)
        ax2.plot(
            snr.reshape(-1),
            res["Linear_TIN"].reshape(-1),
            label="Linear, b=" + str(i),
            linewidth=2,
            marker = markers[0],
            linestyle = linestyes[0],
            color= colors[ind],
        )
                        
        ax2.plot(
            snr.reshape(-1),
            res["Learned_TIN"].reshape(-1),
            label="GD, b=" + str(i),
            linewidth=2,
            marker = markers[1],
            linestyle = linestyes[1],
            color= colors[ind],
        )
        ax2.plot(
            snr.reshape(-1),
            res["Gaussian_TIN"].reshape(-1),
            label="Gaussian, b=" + str(i),
            linewidth=2,
            marker = markers[2],
            linestyle = linestyes[2],
            color= colors[ind],
        )
       
        
        pdfx = io.loadmat(folder + subfolder + str(i) + subfolder_2 + high_power_folder + "/pdf.mat")
        if np.sum(pdfx['RX1_ki'] <0) > 0:
            pdfx['RX1_ki'] = torch.relu(torch.tensor(pdfx['RX1_ki']))
            
        alphx = io.loadmat(folder + subfolder + str(i) + subfolder_2 + high_power_folder + "/alph.mat")
        ax3.plot(alphx['RX1_tin'].reshape(-1),
            pdfx['RX1_tin'].reshape(-1),
            label="b=" + str(i),
            linewidth=2,
            color= colors[ind],
            marker = markers[ind],
            linestyle = linestyes[ind],
        )
        ax4.plot(alphx['RX1_ki'].reshape(-1),
            pdfx['RX1_ki'].reshape(-1),
            label="b=" + str(i),
            linewidth=2,
            color= colors[ind],
            marker = markers[ind],
            linestyle = linestyes[ind],
        )
        ind += 1
            
    
        
    ax1.legend(loc="best", fontsize=8)
    ax1.set_xlabel("SNR (dB)")
    ax1.set_ylabel("Rate User 1")
    ax1.set_title("KI with X2 fixed with INR: " + str(inr))
    ax1 = grid_minor(ax1)
    os.makedirs(folder + "/Plots", exist_ok=True)
    fig1.savefig(folder + "/Plots/KI_X2_fixed.png", bbox_inches="tight")
    plt.close()
    
    ax2.legend(loc="best", fontsize=8)
    ax2.set_xlabel("SNR (dB)")
    ax2.set_ylabel("Rate User 1")
    ax2.set_title("TIN with X2 fixed with INR: " + str(inr))
    ax2 = grid_minor(ax2)
    fig2.savefig(folder + "/Plots/TIN_X2_fixed.png", bbox_inches="tight")
    plt.close()
    
    
    ax3.legend(loc="best", fontsize=8)
    ax3.set_xlabel("X1")
    ax3.set_ylabel("PDF")
    ax3.set_title("TIN INR: " + str(round(inr)) + " SNR: " + str(round(fixed_snr)))
    ax3 = grid_minor(ax3)
    fig3.savefig(folder + "/Plots/TIN_PDF_X1_fixed.png", bbox_inches="tight")
    plt.close()
    
    ax4.legend(loc="best", fontsize=8)
    ax4.set_xlabel("X1")
    ax4.set_ylabel("PDF")
    ax4.set_title("KI INR: " + str(round(inr)) + " SNR: " + str(round(fixed_snr)))
    ax4 = grid_minor(ax4)
    fig4.savefig(folder + "/Plots/KI_PDF_X1_fixed.png", bbox_inches="tight")
    plt.close()
   
    
    print("Plot saved in ", folder + "Plots")

    

def main():
    plt.rcParams.update({"font.size": 14})
    plt.rcParams["text.usetex"] = True

    plot_bits_tin_ki_x2_fixed()
    # main_folder = "Paper_Figure/Regime 3 Hardware/Int/New/BothOptimize-NoADC/"
    # sub_folder = (
    #     main_folder
    #     + "Avg_hw_gain=15.83_iip3=-6.3_bw=500000000_NF1=4.53_NF2=15_regime=3_N=100_max_iter=10000_lr=[0.0001]_pw1_fixed_SNR=5_pw2=3.5727283815192956e-17_a=1_k=1.082642326745063e-05_k2=1.082642326745063e-05"
    # )
    # iip3 = -6.3

    # # plot_R1_R2_curve_for_different_power(main_folder, sub_folder, iip3)
    # range_power = 1e-18
    # # plot_low_power_rate_curve(main_folder, range_power)
    # plot_change_over_pw_FixedX2()


if __name__ == "__main__":
    main()
