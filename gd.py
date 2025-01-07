import torch
from utils import (
    project_pdf,
    loss,
    return_regime_class,
    get_interference_alphabet_x_y,
    loss_interference,
    get_alphabet_x_y,
    check_pdf_x_region,
    get_regime_class_interference,
)
import numpy as np
import copy
from blahut_arimoto_capacity import (
    apply_blahut_arimoto,
    return_pdf_y_x_for_ba,
    blahut_arimoto_torch,
)

from First_Regime import First_Regime


def gd_capacity(config, power, regime_class):
    print("------GD Capacity Calculation--------")
    max_capacity = 0
    max_dict = {}
    max_opt_capacity = torch.tensor([])
    max_pdf_x = torch.tensor([])
    max_alphabet_x = torch.tensor([])
    count_no_impr = 0

    for lr in config["lr"]:

        # alphabet_x, alphabet_y, max_x, max_y = get_alphabet_x_y(config, power)
        alphabet_x = regime_class.alphabet_x

        print("Number of Mass Points:", len(alphabet_x))
        if config["cons_type"] == 0 and config["gd_initial_ba"]:
            _, pdf_x = apply_blahut_arimoto(regime_class, config)
            # breakpoint()
            pdf_x = torch.tensor(pdf_x).float()
        else:  # uniform distribution if no initial distribution
            # pdf_x = torch.ones_like(alphabet_x) * 1 / len(alphabet_x)
            # Gaussian distribution
            pdf_x = (
                1
                / (torch.sqrt(torch.tensor([2 * torch.pi * power])))
                * torch.exp(-0.5 * ((alphabet_x) ** 2) / power).float()
            )

            pdf_x = (pdf_x / torch.sum(pdf_x)).to(torch.float32)
        if len(alphabet_x) == 0:
            breakpoint()
        pdf_x = project_pdf(pdf_x, config["cons_type"], alphabet_x, power)
        pdf_x.requires_grad = True

        # optimizer = torch.optim.Adam([pdf_x], lr=lr)
        optimizer = torch.optim.AdamW([pdf_x], lr=lr, weight_decay=1e-5)
        opt_capacity = []

        for i in range(config["max_iter"]):
            # breakpoint()
            # project back
            optimizer.zero_grad()

            loss_it = loss(pdf_x, regime_class, project_active=True)
            # breakpoint()
            loss_it.backward()
            optimizer.step()

            cap = loss_it.detach().clone()
            opt_capacity.append(-cap.detach().numpy())

            if i % 100 == 0:
                print("Iter:", i, "Capacity:", opt_capacity[-1])

            # moving average of capacity between last 100 iterations did not improve, stop
            if not config["gd_nostop_cond"]:
                if (
                    i > 100
                    and np.abs(
                        np.mean(opt_capacity[-50:]) - np.mean(opt_capacity[-100:-50])
                    )
                    < config["epsilon"]
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

    max_pdf_x = project_pdf(max_pdf_x, config["cons_type"], max_alphabet_x, power)

    return max_capacity, max_pdf_x, max_alphabet_x, opt_capacity


def gd_on_alphabet_capacity(max_x, config, power, regime_class):
    if config["cons_type"] != 0:
        raiseValueError("Constraint type should be Peak Power for this function")

    print("------GD over Alphabet Capacity Calculation--------")
    max_capacity = 0
    max_dict = {}
    max_opt_capacity = torch.tensor([])
    max_pdf_x = torch.tensor([])
    max_alphabet_x = torch.tensor([])
    count_no_impr = 0

    for mass_points in config["mass_points"]:
        for lr in config["lr"]:

            print("Number of Mass Points:", mass_points)

            # initial alphabet_x is uniform
            alphabet_x = torch.linspace(-max_x, max_x, mass_points)
            # regime_class.set_alphabet_x(alphabet_x)
            # pdf_y_given_x = return_pdf_y_x_for_ba(config, regime_class)

            # pdf_x = blahut_arimoto_torch(pdf_y_given_x, alphabet_x, log_base=np.e)
            # pdf_x = torch.ones_like(alphabet_x) / mass_points

            alphabet_x.requires_grad = True

            optimizer = torch.optim.Adam([alphabet_x], lr=lr)

            opt_capacity = []

            for i in range(config["max_iter"]):
                # project back
                optimizer.zero_grad()

                loss_it, pdf_x = loss_alphabet_x(
                    alphabet_x, config, regime_class, max_x
                )
                try:
                    # loss_it.requires_grad = True
                    loss_it.backward()
                except:
                    breakpoint()
                if torch.sum(alphabet_x.isnan()) > 0:
                    breakpoint()
                optimizer.step()

                cap = loss_it.detach().clone()
                opt_capacity.append(-cap.detach().numpy())
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

    return max_capacity, max_pdf_x, max_alphabet_x


def loss_alphabet_x(alphabet_x, config, regime_class, max_x):
    if torch.sum(alphabet_x.isnan()) > 0:
        breakpoint()
    # Project back to the feasible set
    alphabet_x = torch.clip(alphabet_x, -max_x, max_x)

    # TODO: This does not work for regime 2
    # pdf_y_given_x = return_pdf_y_x_for_ba(config, regime_class)

    # breakpoint()
    # pdf_x = torch.ones_like(alphabet_x) / torch.tensor([config["mass_points"]])
    # pdf_x = pdf_x[0]

    # currently only tanh is supported -  Regime 1
    alphabet_v = torch.tanh(alphabet_x)
    pdf_y_given_x = regime_class.calculate_pdf_y_given_v(alphabet_v)
    pdf_y_given_x = torch.transpose(pdf_y_given_x, 0, 1)

    # pdf_x = blahut_arimoto_torch(pdf_y_given_x, alphabet_x, log_base=np.e)
    pdf_x = torch.ones_like(alphabet_x) / len(alphabet_x)
    if torch.sum(pdf_x.isnan()) > 0:
        breakpoint()
    # breakpoint()
    # pdf_y_given_x = torch.transpose(pdf_y_given_x, 0, 1)
    # pdf_x = torch.ones_like(alphabet_x) / len(alphabet_x)
    pdf_x_given_y = pdf_y_given_x * pdf_x
    pdf_x_given_y = torch.transpose(pdf_x_given_y, 0, 1) / torch.sum(
        pdf_x_given_y, axis=1
    )
    c = 0
    for i in range(len(alphabet_x)):
        if pdf_x[i] > 0:
            c += torch.sum(
                pdf_x[i]
                * pdf_y_given_x[:, i]
                * torch.log(pdf_x_given_y[i, :] / pdf_x[i] + 1e-16)
            )
    c = c
    loss = -c
    if torch.isnan(loss):
        breakpoint()
    if torch.sum(pdf_x.isnan()) > 0:
        breakpoint()

    return loss, pdf_x


def gradient_alphabet_lambda_loss():

    pass


def gradient_descent_on_interference(config, power, power2, lambda_sweep):
    # It should return R1 and R2 pairs for different lambda values
    # The loss function is lambda*Rate1 + (1-lambda)*Rate2

    # In the main loop, we can plot R1-R2 pairs by marking the gaussian for that power level in the graph

    # FIXME: Currently only works for regime 1 -- So I directly assume instead of getting it as an input
    print("---Gradient Descent on Interference Channel ----")

    if config["regime"] != 1 and config["regime"] != 3:
        raise ValueError("This function only works for regime 1 or regime 3")

    alphabet_x_RX1, alphabet_y_RX1, alphabet_x_RX2, alphabet_y_RX2 = (
        get_interference_alphabet_x_y(config, power, power2)
    )

    reg_RX1, reg_RX2 = get_regime_class_interference(
        alphabet_x_RX1,
        alphabet_x_RX2,
        alphabet_y_RX1,
        alphabet_y_RX2,
        config,
        power,
        power2,
    )

    # Initializations
    max_sum_cap = []
    max_pdf_x_RX1 = []
    max_pdf_x_RX2 = []
    max_cap_RX1 = []
    max_cap_RX2 = []
    save_opt_sum_capacity = []

    if config["x2_fixed"]:
        pdf_x_RX2 = get_fixed_interferer(config, alphabet_x_RX2, alphabet_y_RX2)

    for ind, lmbd in enumerate(lambda_sweep):
        # FIXME: currently different learning rate comparison is not supported
        print("++++++++ Lambda: ", lmbd, " ++++++++")

        for lr in config["lr"]:

            # Initial distributions are uniform for peak power constraint
            # if config["cons_type"] == 0:
            pdf_x_RX1 = torch.ones_like(alphabet_x_RX1) * 1 / len(alphabet_x_RX1)
            pdf_x_RX1 = pdf_x_RX1 / torch.sum(pdf_x_RX1)
            pdf_x_RX1 = project_pdf(
                pdf_x_RX1, config["cons_type"], alphabet_x_RX1, power
            )
            pdf_x_RX1.requires_grad = True

            if not config["x2_fixed"]:
                pdf_x_RX2 = torch.ones_like(alphabet_x_RX2) * 1 / len(alphabet_x_RX2)
                pdf_x_RX2 = pdf_x_RX2 / torch.sum(pdf_x_RX2)
                pdf_x_RX2 = project_pdf(
                    pdf_x_RX2, config["cons_type"], alphabet_x_RX2, config["power_2"]
                )
                pdf_x_RX2.requires_grad = True

                optimizer = torch.optim.Adam([pdf_x_RX1, pdf_x_RX2], lr=lr)
            else:
                optimizer = torch.optim.Adam([pdf_x_RX1], lr=lr)

            opt_sum_capacity = []
            max_sum_cap_h = 0

            for i in range(config["max_iter"]):
                optimizer.zero_grad()
                if torch.sum(pdf_x_RX1.isnan()) > 0 or torch.sum(pdf_x_RX2.isnan()) > 0:
                    breakpoint()

                loss, cap_RX1, cap_RX2 = loss_interference(
                    pdf_x_RX1, pdf_x_RX2, reg_RX1, reg_RX2, lmbd
                )

                loss.backward()
                optimizer.step()
                sum_capacity = loss.detach().clone()
                opt_sum_capacity.append(-sum_capacity.detach().numpy())

                if i % 100 == 0:
                    print(
                        "Iter:",
                        i,
                        " Sum Capacity:",
                        opt_sum_capacity[-1],
                        " R1:",
                        cap_RX1,
                        " R2:",
                        cap_RX2,
                    )
                if opt_sum_capacity[-1] > max_sum_cap_h:
                    max_sum_cap_h = opt_sum_capacity[-1]
                    max_pdf_x_RX1_h = pdf_x_RX1.clone().detach()
                    max_pdf_x_RX2_h = pdf_x_RX2.clone().detach()
                    max_cap_RX1_h = cap_RX1.clone().detach().numpy()
                    max_cap_RX2_h = cap_RX2.clone().detach().numpy()

                if (
                    i > 100
                    and np.abs(
                        np.mean(opt_sum_capacity[-50:])
                        - np.mean(opt_sum_capacity[-100:-50])
                    )
                    < config["epsilon"]
                ):
                    break

            save_opt_sum_capacity.append(opt_sum_capacity)
            max_sum_cap.append(max_sum_cap_h)
            max_cap_RX1.append(max_cap_RX1_h)
            max_cap_RX2.append(max_cap_RX2_h)

            # save the pdfs after projection
            pdf_x_RX1 = project_pdf(
                max_pdf_x_RX1_h, config["cons_type"], alphabet_x_RX1, power
            )
            pdf_x_RX2 = project_pdf(
                max_pdf_x_RX2_h, config["cons_type"], alphabet_x_RX2, config["power_2"]
            )
            max_pdf_x_RX1.append(pdf_x_RX1.detach().clone().numpy())
            max_pdf_x_RX2.append(pdf_x_RX2.detach().clone().numpy())

            print(
                "*****Max Capacity:",
                max_sum_cap_h,
                "R1:",
                max_cap_RX1_h,
                "R2:",
                max_cap_RX2_h,
                "*****",
            )
    return (
        max_sum_cap,
        max_pdf_x_RX1,
        max_pdf_x_RX2,
        max_cap_RX1,
        max_cap_RX2,
        save_opt_sum_capacity,
    )


# !! CANCELED
def sequential_gradient_descent_on_interference(config, power, power2, lambda_sweep):
    # It should return R1 and R2 pairs for different lambda values
    # The loss function is lambda*Rate1 + (1-lambda)*Rate2

    # In the main loop, we can plot R1-R2 pairs by marking the gaussian for that power level in the graph

    print("---Sequential Gradient Descent on Interference Channel  ----")

    if config["regime"] != 1:
        raise ValueError("This function only works for regime 1")

    alphabet_x_RX1, alphabet_y_RX1, alphabet_x_RX2, alphabet_y_RX2 = (
        get_interference_alphabet_x_y(config, power, power2)
    )

    # FIXME : This is not clean
    config["sigma_2"] = config["sigma_22"]

    f_reg_RX2 = First_Regime(alphabet_x_RX2, alphabet_y_RX2, config, power)
    config["sigma_2"] = config["sigma_12"]
    f_reg_RX1 = First_Regime(alphabet_x_RX1, alphabet_y_RX1, config, power)

    # Initializations
    max_sum_cap = []
    max_pdf_x_RX1 = []
    max_pdf_x_RX2 = []
    max_cap_RX1 = []
    max_cap_RX2 = []
    save_opt_sum_capacity = []

    for ind, lmbd in enumerate(lambda_sweep):
        # FIXME: currently different learning rate comparison is not supported
        print("++++++++ Lambda: ", lmbd, " ++++++++")

        for lr in config["lr"]:
            # Both RX1 and RX2 will have the same delta separation between points - number of mass points dont matter
            # FIXME: Change the current implementation to this as well- it makes more sense I think

            # Initial distributions are uniform for peak power constraint
            if config["cons_type"] == 0:
                pdf_x_RX1 = torch.ones_like(alphabet_x_RX1) * 1 / len(alphabet_x_RX1)
                pdf_x_RX2 = torch.ones_like(alphabet_x_RX2) * 1 / len(alphabet_x_RX2)
            elif config["cons_type"] == 1:
                # # Initial distributions are gaussian for average power constraint
                pdf_x_RX1 = (
                    1
                    / (torch.sqrt(torch.tensor([2 * torch.pi]) * power))
                    * torch.exp(-0.5 * ((alphabet_x_RX1) ** 2) / power)
                )

                pdf_x_RX1 = project_pdf(
                    pdf_x_RX1, config["cons_type"], alphabet_x_RX1, power
                )

                pdf_x_RX2 = (
                    1
                    / (torch.sqrt(torch.tensor([2 * torch.pi]) * power))
                    * torch.exp(-0.5 * ((alphabet_x_RX2) ** 2) / power)
                )
                pdf_x_RX2 = project_pdf(
                    pdf_x_RX2, config["cons_type"], alphabet_x_RX2, power
                )
            else:
                raise ValueError("Constraint type not supported")
            opt_sum_capacity = []
            seq_cap = []
            pdf_x_RX1.requires_grad = True
            pdf_x_RX2.requires_grad = True
            optimizer_RX2 = torch.optim.AdamW([pdf_x_RX2], lr=lr, weight_decay=1e-5)
            # optimizer_RX2 = torch.optim.Adam([pdf_x_RX2], lr=lr)
            optimizer_RX1 = torch.optim.AdamW([pdf_x_RX1], lr=lr, weight_decay=1e-5)
            # optimizer_RX1 = torch.optim.Adam([pdf_x_RX1], lr=lr)
            for sq in range(config["max_seq_iter"]):
                # Now optimize for RX1
                for i in range(config["max_iter"]):
                    optimizer_RX1.zero_grad()

                    loss, cap_RX1, cap_RX2 = loss_interference(
                        pdf_x_RX1,
                        pdf_x_RX2,
                        f_reg_RX1,
                        f_reg_RX2,
                        lmbd,
                        upd_RX2=False,
                        upd_RX1=True,
                    )

                    loss.backward()
                    optimizer_RX1.step()
                    sum_capacity = loss.detach().clone()
                    opt_sum_capacity.append(-sum_capacity.detach().numpy())
                    # FIXME: projection moved here
                    # pdf_x_RX1 = project_pdf(
                    #     pdf_x_RX1,
                    #     f_reg_RX1.config["cons_type"],
                    #     f_reg_RX1.alphabet_x,
                    #     f_reg_RX1.power,
                    # )

                    if i % 100 == 0:
                        print(
                            "Iter of RX1 update:",
                            i,
                            " Sum Capacity:",
                            opt_sum_capacity[-1],
                            " R1:",
                            cap_RX1,
                            " R2:",
                            cap_RX2,
                        )

                    if (
                        i > 100
                        and np.abs(
                            np.mean(opt_sum_capacity[-50:])
                            - np.mean(opt_sum_capacity[-100:-50])
                        )
                        < 1e-5
                    ):
                        break
                pdf_x_RX1 = project_pdf(
                    pdf_x_RX1, config["cons_type"], alphabet_x_RX1, power
                )

                # Optimize for RX2
                for i in range(config["max_iter"]):
                    optimizer_RX2.zero_grad()

                    loss, cap_RX1, cap_RX2 = loss_interference(
                        pdf_x_RX1,
                        pdf_x_RX2,
                        f_reg_RX1,
                        f_reg_RX2,
                        lmbd,
                        upd_RX1=False,
                        upd_RX2=True,
                    )

                    loss.backward()
                    optimizer_RX2.step()
                    sum_capacity = loss.detach().clone()
                    opt_sum_capacity.append(-sum_capacity.detach().numpy())
                    # FIXME: projection moved here
                    # pdf_x_RX2 = project_pdf(
                    #     pdf_x_RX2,
                    #     f_reg_RX2.config["cons_type"],
                    #     f_reg_RX2.alphabet_x,
                    #     f_reg_RX2.power,
                    # )
                    if i % 100 == 0:
                        print(
                            "Iter of RX2 update:",
                            i,
                            " Sum Capacity:",
                            opt_sum_capacity[-1],
                            " R1:",
                            cap_RX1,
                            " R2:",
                            cap_RX2,
                        )

                    if (
                        i > 100
                        and np.abs(
                            np.mean(opt_sum_capacity[-50:])
                            - np.mean(opt_sum_capacity[-100:-50])
                        )
                        < 1e-5
                    ):
                        break

                # RX2 pdf projection
                pdf_x_RX2 = project_pdf(
                    pdf_x_RX2, config["cons_type"], alphabet_x_RX2, config["power_2"]
                )

                seq_cap.append(opt_sum_capacity[-1])
                if sq > 1 and np.abs(seq_cap[-1] - seq_cap[-2]) < 1e-5:
                    break

            save_opt_sum_capacity.append(opt_sum_capacity)
            max_sum_cap.append(seq_cap)
            max_cap_RX1.append(cap_RX1.detach().numpy())
            max_cap_RX2.append(cap_RX2.detach().numpy())

            pdf_x_RX1 = project_pdf(
                pdf_x_RX1, config["cons_type"], alphabet_x_RX1, power
            )

            pdf_x_RX2 = project_pdf(
                pdf_x_RX2, config["cons_type"], alphabet_x_RX2, config["power_2"]
            )

            max_pdf_x_RX1.append(pdf_x_RX1.detach().clone().numpy())
            max_pdf_x_RX2.append(pdf_x_RX2.detach().clone().numpy())

    # breakpoint()
    return (
        max_sum_cap,
        max_pdf_x_RX1,
        max_pdf_x_RX2,
        max_cap_RX1,
        max_cap_RX2,
        save_opt_sum_capacity,
    )


# DID NOT FINISH
def gradient_descent_projection_with_learning_rate(config, power, power2, lambda_sweep):
    print("---Gradient Descent on Interference Channel While X2 is fixed----")

    if config["regime"] != 1:
        raise ValueError("This function only works for regime 1")

    alphabet_x_RX1, alphabet_y_RX1, alphabet_x_RX2, alphabet_y_RX2 = (
        get_interference_alphabet_x_y(config, power, power2)
    )

    # FIXME : This is not clean
    config["sigma_2"] = config["sigma_22"]
    f_reg_RX2 = First_Regime(alphabet_x_RX2, alphabet_y_RX2, config, config["power_2"])
    config["sigma_2"] = config["sigma_12"]
    f_reg_RX1 = First_Regime(alphabet_x_RX1, alphabet_y_RX1, config, power)

    # Initializations
    max_sum_cap = []
    max_pdf_x_RX1 = []
    max_pdf_x_RX2 = []
    max_cap_RX1 = []
    max_cap_RX2 = []
    save_opt_sum_capacity = []

    for ind, lmbd in enumerate(lambda_sweep):
        # FIXME: currently different learning rate comparison is not supported
        print("++++++++ Lambda: ", lmbd, " ++++++++")

        for lr in config["lr"]:
            # Both RX1 and RX2 will have the same delta separation between points - number of mass points dont matter
            # FIXME: Change the current implementation to this as well- it makes more sense I think

            pdf_x_RX1 = torch.ones_like(alphabet_x_RX1) * 1 / len(alphabet_x_RX1)
            pdf_x_RX2 = torch.ones_like(alphabet_x_RX2) * 1 / len(alphabet_x_RX2)

            pdf_x_RX1 = project_pdf(
                pdf_x_RX1, config["cons_type"], alphabet_x_RX1, power
            )

            pdf_x_RX2 = project_pdf(
                pdf_x_RX2, config["cons_type"], alphabet_x_RX2, config["power_2"]
            )

            pdf_x_RX1.requires_grad = True
            pdf_x_RX2.requires_grad = True

            opt_sum_capacity = []
            max_sum_cap_h = 0
            for i in range(config["max_iter"]):
                pdf_x_RX1.grad = torch.ones_like(pdf_x_RX1)
                pdf_x_RX2.grad = torch.ones_like(pdf_x_RX2)

                loss, cap_RX1, cap_RX2 = loss_interference(
                    pdf_x_RX1, pdf_x_RX2, f_reg_RX1, f_reg_RX2, lmbd
                )

                loss.backward()
                sum_capacity = loss.detach().clone()
                opt_sum_capacity.append(-sum_capacity.detach().numpy())
                with torch.no_grad():
                    pdf_x_RX1_grad = pdf_x_RX1.grad
                    pdf_x_RX1 -= lr * pdf_x_RX1_grad

                    check = 0
                    lr_new = lr
                    enough_check = True

                    while (
                        not check_pdf_x_region(
                            pdf_x_RX1, alphabet_x_RX1, config["cons_type"], power
                        )
                    ) and enough_check:
                        lr_new = lr_new / 2
                        pdf_x_RX1 += lr_new * pdf_x_RX1_grad

                        check += 1
                        if check > config["max_lr_check"]:
                            enough_check = False

                    pdf_x_RX2_grad = pdf_x_RX2.grad
                    pdf_x_RX2 -= lr * pdf_x_RX2_grad
                    check = 0
                    lr_new = lr
                    enough_check = True

                    while (
                        not check_pdf_x_region(
                            pdf_x_RX2,
                            alphabet_x_RX2,
                            config["cons_type"],
                            config["power_2"],
                        )
                    ) and enough_check:
                        lr_new = lr_new / 2
                        pdf_x_RX2 += lr_new * pdf_x_RX2_grad
                        check += 1
                        if check > 20:
                            enough_check = False
                if i % 100 == 0:
                    print(
                        "Iter:",
                        i,
                        " Sum Capacity:",
                        opt_sum_capacity[-1],
                        " R1:",
                        cap_RX1,
                        " R2:",
                        cap_RX2,
                    )
                if opt_sum_capacity[-1] > max_sum_cap_h:
                    max_sum_cap_h = opt_sum_capacity[-1]
                    max_pdf_x_RX1_h = pdf_x_RX1.clone().detach()
                    max_pdf_x_RX2_h = pdf_x_RX2.clone().detach()
                    max_cap_RX1_h = cap_RX1.clone().detach().numpy()
                    max_cap_RX2_h = cap_RX2.clone().detach().numpy()

                if (
                    i > 100
                    and np.abs(
                        np.mean(opt_sum_capacity[-50:])
                        - np.mean(opt_sum_capacity[-100:-50])
                    )
                    < config["epsilon"]
                ):
                    break

            save_opt_sum_capacity.append(opt_sum_capacity)
            max_sum_cap.append(max_sum_cap_h)
            max_cap_RX1.append(max_cap_RX1_h)
            max_cap_RX2.append(max_cap_RX2_h)

            # save the pdfs after projection
            pdf_x_RX1 = project_pdf(
                max_pdf_x_RX1_h, config["cons_type"], alphabet_x_RX1, power
            )
            pdf_x_RX2 = project_pdf(
                max_pdf_x_RX2_h, config["cons_type"], alphabet_x_RX2, config["power_2"]
            )
            max_pdf_x_RX1.append(pdf_x_RX1.detach().clone().numpy())
            max_pdf_x_RX2.append(pdf_x_RX2.detach().clone().numpy())

            print(
                "*****Max Capacity:",
                max_sum_cap_h,
                "R1:",
                max_cap_RX1_h,
                "R2:",
                max_cap_RX2_h,
                "*****",
            )
    # breakpoint()
    return (
        max_sum_cap,
        max_pdf_x_RX1,
        max_pdf_x_RX2,
        max_cap_RX1,
        max_cap_RX2,
        save_opt_sum_capacity,
    )


def get_fixed_interferer(config, alphabet_x_RX2, alphabet_y_RX2):
    if config["x2_type"] == 0:
        print(" +++----- X2 Distribution is Gaussian ------ +++")
        pdf_x_RX2 = (
            1
            / (torch.sqrt(torch.tensor([2 * torch.pi * config["power_2"]])))
            * torch.exp(-0.5 * ((alphabet_x_RX2) ** 2) / config["power_2"]).float()
        )
    elif config["x2_type"] == 1:
        print(" +++----- X2 Distribution is calculated ------ +++")
        config["sigma_1"] = config["sigma_21"]
        config["sigma_2"] = config["sigma_22"]

        regime_class = return_regime_class(
            config, alphabet_x_RX2, alphabet_y_RX2, config["power_2"]
        )
        config["gd_nostop_cond"] = True
        _, pdf_x_RX2, _, _ = gd_capacity(config, config["power_2"], regime_class)

    else:
        raise ValueError("Interferer type not defined")
    pdf_x_RX2 = (pdf_x_RX2 / torch.sum(pdf_x_RX2)).to(torch.float32)
    pdf_x_RX2 = project_pdf(
        pdf_x_RX2, config["cons_type"], alphabet_x_RX2, config["power_2"]
    )
    # why pdf_x_RX2 decreases after this?????
    return pdf_x_RX2


def main():
    pass


if __name__ == "__main__":
    main()
