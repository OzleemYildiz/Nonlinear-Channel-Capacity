import torch
from utils import project_pdf, loss, capacity
import numpy as np


def gd_capacity(max_x, alphabet_y, config, power):
    print("GD Capacity Calculation")
    max_capacity = 0
    max_dict = {}
    max_opt_capacity = torch.tensor([])
    max_pdf_x = torch.tensor([])
    max_alphabet_x = torch.tensor([])
    count_no_impr = 0

    for i in range(config["max_mass_points"]):
        for lr in config["lr"]:
            mass_points = i + 100

            print("Number of Mass Points:", mass_points)

            alphabet_x = torch.linspace(-max_x, max_x, mass_points)
            pdf_x = (
                torch.ones_like(alphabet_x) * 1 / mass_points
            )  # (uniform distribution)

            pdf_x.requires_grad = True

            optimizer = torch.optim.Adam([pdf_x], lr=lr)

            opt_capacity = []

            for i in range(config["max_iter"]):
                # project back
                optimizer.zero_grad()
                h_y_negative = loss(
                    pdf_x,
                    alphabet_y,
                    alphabet_x,
                    power,
                    config,
                )
                cap = capacity(
                    h_y_negative, config, alphabet_x, pdf_x, power, alphabet_y
                )
                loss_it = cap * (-1)
                loss_it.backward()
                optimizer.step()

                opt_capacity.append(cap.detach().numpy())

                if i % 100 == 0:
                    print("Iter:", i, "Capacity:", opt_capacity[-1])

                # moving average of capacity between last 100 iterations did not improve, stop
                if (
                    i > 100
                    and np.abs(
                        np.mean(opt_capacity[-50:]) - np.mean(opt_capacity[-100:-50])
                    )
                    < 1e-5
                ):
                    break

            # is it enough mass point check?
            if opt_capacity[-1] > max_capacity:
                count_no_impr = 0  # improvement happened

                # p_pdf_x = project_pdf(pdf_x, cons_type, max_alphabet_x, power)

                max_opt_capacity = opt_capacity
                max_pdf_x = pdf_x
                max_alphabet_x = alphabet_x

                if np.abs(opt_capacity[-1] - max_capacity) < config["epsilon"]:
                    break
                max_capacity = opt_capacity[-1]
            else:
                count_no_impr += 1

            if count_no_impr > config["max_k_nochange"]:
                break

    max_pdf_x = project_pdf(max_pdf_x, config["cons_type"], max_alphabet_x, power)

    return max_capacity, max_pdf_x
