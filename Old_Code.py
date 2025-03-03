# Following is omitted from main.py - Currently not necessary features

# FIXME: This condition might not work right now
if config["ba_active"] and config["cons_type"] == 0:
    (cap, input_dist) = apply_blahut_arimoto(regime_class, config)
    loss_g = loss(torch.tensor(input_dist).to(torch.float32), regime_class)
    if ind == 0:  # keeping record of only tau results for demonstration
        capacity_ba.append(-loss_g)
    if not config["time_division_active"]:
        map_pdf_ba[str(snr)] = [
            input_dist,
            regime_class.alphabet_x.numpy(),
        ]
    else:
        map_pdf_ba[str(tau)] = [
            input_dist,
            regime_class.alphabet_x.numpy(),
        ]

# If constraint type is 2, calculate gaussian with optimized snr  -- Ruth's paper
# FIXME: Might not be working
if config["cons_type"] == 2:
    cap_r = gaussian_with_l1_norm(alphabet_x, alphabet_y, power, config)
    capacity_ruth.append(cap_r)
