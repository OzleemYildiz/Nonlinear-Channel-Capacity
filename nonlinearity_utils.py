import torch
import numpy as np


# It is one-one function so pdf stays the alphabet changes
def alphabet_fix_nonlinear(alphabet_x, config):
    if config["regime"] == 1:
        nonlinear_func = return_nonlinear_fn(config)
        x_change = nonlinear_func(alphabet_x)
    elif config["regime"] == 2:
        x_change = alphabet_x  # since nonlinearity is applied to the channel output
    return x_change


def return_nonlinear_fn(config, tanh_factor=None):
    # 0:linear
    if config["nonlinearity"] == 0:
        nonlinear_fn = lambda x: x
    #  1:nonlinear C1 (sgn(X)sqrt(abs(X)))
    elif config["nonlinearity"] == 1:
        nonlinear_fn = lambda x: torch.sign(torch.tensor(x)) * torch.sqrt(
            torch.abs(torch.tensor(x))
        )
    # 2: nonlinear C2 (sgn(X)abs(X)^{0.9})
    elif config["nonlinearity"] == 2:
        nonlinear_fn = (
            lambda x: torch.sign(torch.tensor(x)) * torch.abs(torch.tensor(x)) ** 0.9
        )
    # 3:nonlinear tanh(X)
    elif config["nonlinearity"] == 3:
        if tanh_factor is None:
            raise ValueError("Tanh factor not defined")
        nonlinear_fn = lambda x: tanh_factor * torch.tanh(torch.tensor(x / tanh_factor))
    # 4:nonlinear x4: x/(1 + x^4)^1/4
    elif config["nonlinearity"] == 4:
        nonlinear_fn = lambda x: torch.tensor(x / ((1 + x**4) ** (1 / 4)))
    # 5:nonlinear clipping
    elif config["nonlinearity"] == 5:
        # nonlinear_fn = lambda x: torch.ones_like(torch.tensor(x))
        nonlinear_fn = (
            lambda x: config["clipping_limit_y"]
            / config["clipping_limit_x"]
            * torch.clip(
                torch.tensor(x), -config["clipping_limit_x"], config["clipping_limit_x"]
            )
        )
    else:
        raise ValueError("Nonlinearity not defined")
    return nonlinear_fn

#NOT USED
def return_derivative_of_nonlinear_fn(config, tanh_factor=None):
    # This function is numpy

    # 0:linear
    if config["nonlinearity"] == 0:
        nonlinear_fn = lambda x: 1
    elif config["nonlinearity"] == 1:
        nonlinear_fn = lambda x: (
            np.sign(x) / (2 * np.sqrt(np.abs(x))) if abs(x) > 1e-3 else 50
        )
    # 3:nonlinear tanh(X)
    elif config["nonlinearity"] == 3:
        if tanh_factor is None:
            raise ValueError("Tanh factor not defined")
        nonlinear_fn = lambda x: (1 / np.cosh(x / tanh_factor)) ** 2
    # 4:nonlinear x4: x/(1 + x^4)^1/4
    elif config["nonlinearity"] == 4:
        nonlinear_fn = lambda x: 1 / (1 + x**4) ** (5 / 4)
    elif config["nonlinearity"] == 5:
        nonlinear_fn = lambda x: (
            config["clipping_limit_y"] / config["clipping_limit_x"]
            if -config["clipping_limit_x"] < x < config["clipping_limit_x"]
            else 0
        )

    else:
        raise ValueError("Derivative is not supported")

    return nonlinear_fn


def return_nonlinear_fn_numpy(config, tanh_factor=None):    
    # 0:linear
    if config["nonlinearity"] == 0:
        nonlinear_fn = lambda x: x
    #  1:nonlinear C1 (sgn(X)sqrt(abs(X)))
    elif config["nonlinearity"] == 1:
        nonlinear_fn = lambda x: np.sign(x) * np.sqrt(np.abs(x))
    # 2: nonlinear C2 (sgn(X)abs(X)^{0.9})
    elif config["nonlinearity"] == 2:
        nonlinear_fn = lambda x: np.sign(x) * np.abs(x) ** 0.9
    # 3:nonlinear tanh(X)
    elif config["nonlinearity"] == 3:
        if tanh_factor is None:
            raise ValueError("Tanh factor not defined")
        nonlinear_fn = lambda x: tanh_factor * np.tanh(
            x / tanh_factor
        )
    # 4:nonlinear x4: x/(1 + x^4)^1/4
    elif config["nonlinearity"] == 4:
        nonlinear_fn = lambda x: x / ((1 + x**4) ** (1 / 4))
    elif config["nonlinearity"] == 5:
        nonlinear_fn = lambda x: np.clip(
            x, -config["clipping_limit_x"], config["clipping_limit_x"]
        )
    else:
        raise ValueError("Nonlinearity not defined")

    return nonlinear_fn
