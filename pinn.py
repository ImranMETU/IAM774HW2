"""Soft-constraint PINN for forward heat conduction."""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from heat import relative_l2


def make_model(width: int, hidden_layers: int) -> nn.Sequential:
    layers = []
    features = 2
    for _ in range(hidden_layers):
        layers.extend([nn.Linear(features, width, dtype=torch.float64), nn.Tanh()])
        features = width
    layers.append(nn.Linear(features, 1, dtype=torch.float64))
    return nn.Sequential(*layers)


def sample_points(counts, generator):
    """Draw a fixed full batch, splitting boundary samples evenly across ends."""
    def uniform(n, dims):
        return torch.rand((n, dims), generator=generator, dtype=torch.float64)

    interior = uniform(counts["interior"], 2)
    n = counts["boundary"]
    boundary = torch.cat((torch.zeros(n // 2, 1), torch.ones(n - n // 2, 1)))
    boundary = torch.cat((boundary.to(torch.float64), uniform(n, 1)), dim=1)
    x = uniform(counts["initial"], 1)
    initial = torch.cat((x, torch.zeros_like(x)), dim=1)
    return interior, boundary, initial


def pde_residual(model, points, rod):
    # Separate coordinate leaf avoids accumulating coordinate gradients across steps.
    coords = points.detach().requires_grad_(True)
    u = model(coords)
    du = torch.autograd.grad(u, coords, torch.ones_like(u), create_graph=True)[0]
    ux, ut = du[:, :1], du[:, 1:]
    uxx = torch.autograd.grad(ux, coords, torch.ones_like(ux), create_graph=True)[0][:, :1]
    return ut - rod.kappa * uxx + rod.loss_rate * u


def loss_terms(model, points, initial_target, rod):
    interior, boundary, initial = points
    return {
        "pde": pde_residual(model, interior, rod).square().mean(),
        "bc": model(boundary).square().mean(),
        "ic": (model(initial) - initial_target).square().mean(),
    }


def field_metrics(prediction, exact):
    if prediction.ndim != 2 or prediction.shape != exact.shape:
        raise ValueError("Expected matching time-by-space fields.")
    by_time = np.array([relative_l2(row, truth) for row, truth in zip(prediction, exact)])
    return {
        "relative_l2": relative_l2(prediction, exact),
        "relative_l2_t1": float(by_time[-1]),
        "max_absolute_error": float(np.max(np.abs(prediction - exact))),
    }, by_time


def save_checkpoint(path: Path, model, config):
    torch.save({"model_state": model.state_dict(), "config": config}, path)


def load_checkpoint(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = make_model(**config["model"])
    model.load_state_dict(checkpoint["model_state"])
    return model.eval(), config
