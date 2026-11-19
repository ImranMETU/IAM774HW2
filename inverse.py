"""Synthetic observations and differentiable inverse heat conduction."""

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
from scipy.optimize import least_squares
import torch

from heat import Rod
from pinn import make_model, loss_terms


@dataclass(frozen=True)
class HeatCoefficients:
    kappa: torch.Tensor
    loss_rate: float


def make_observations(exact: Callable, count: int, eta: float, seed: int, t_min: float):
    rng = np.random.default_rng(seed)
    coords = rng.uniform([0, t_min], [1, 1], size=(count, 2))
    clean = exact(coords[:, 0], coords[:, 1])
    sigma = float(eta * np.std(clean, ddof=0))
    noisy = clean + sigma * rng.standard_normal(count)
    return coords, clean, noisy, sigma


def log_diffusivity(kappa_initial: float):
    return torch.nn.Parameter(torch.tensor(math.log(kappa_initial), dtype=torch.float64))


def inverse_terms(model, log_kappa, loss_rate, points, initial_target, coords, observed):
    coeffs = HeatCoefficients(log_kappa.exp(), loss_rate)
    terms = loss_terms(model, points, initial_target, coeffs)
    terms["data"] = (model(coords) - observed).square().mean()
    return terms


def parameter_metrics(estimate, truth):
    relative = estimate / truth - 1
    return {"kappa": float(estimate), "signed_relative_error": float(relative),
            "absolute_relative_error": float(abs(relative))}


def least_squares_reference(coords, observed, loss_rate, amplitude, kappa_initial):
    def residual(log_kappa):
        rod = Rod(float(np.exp(log_kappa[0])), loss_rate, amplitude)
        return rod.exact(coords[:, 0], coords[:, 1]) - observed

    return least_squares(residual, [math.log(kappa_initial)])


def save_inverse_checkpoint(path, model, log_kappa, config):
    torch.save({"model_state": model.state_dict(), "log_kappa": log_kappa.detach(),
                "config": config}, path)


def load_inverse_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = make_model(**checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model_state"])
    return model.eval(), torch.nn.Parameter(checkpoint["log_kappa"]), checkpoint["config"]
