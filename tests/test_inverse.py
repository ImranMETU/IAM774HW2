import numpy as np
import pytest
import torch

from heat import Rod
from inverse import (
    HeatCoefficients, least_squares_reference, load_inverse_checkpoint,
    log_diffusivity, make_observations, parameter_metrics, save_inverse_checkpoint,
)
from lbfgs_diagnostics import MonitoredLBFGS
from pinn import make_model, pde_residual


def test_positive_diffusivity_parameterization():
    s = log_diffusivity(0.10)
    assert s.requires_grad and s.dtype == torch.float64
    assert s.exp().item() == pytest.approx(0.10)
    for value in (-20.0, 0.0, 20.0):
        with torch.no_grad():
            s.fill_(value)
        assert s.exp().item() > 0


def test_observation_sampling_and_noise_scaling():
    rod = Rod(0.03, 0.875, 0.60)
    eta, n, seed = 0.01, 100, 67507
    coords, clean, noisy, sigma = make_observations(rod.exact, n, eta, seed, 0.3)
    rng = np.random.default_rng(seed)
    np.testing.assert_array_equal(coords, rng.uniform([0, 0.3], [1, 1], size=(n, 2)))
    np.testing.assert_allclose(clean, rod.exact(coords[:, 0], coords[:, 1]))
    assert sigma == pytest.approx(eta * np.sqrt(np.mean((clean - clean.mean())**2)))
    np.testing.assert_allclose(noisy, clean + sigma * rng.standard_normal(n))
    np.testing.assert_array_equal(make_observations(rod.exact, n, 0, seed, 0.3)[2], clean)


def test_residual_propagates_diffusivity_gradient():
    s = log_diffusivity(0.10)
    coords = torch.tensor([[0.2, 0.3], [0.5, 0.4]], dtype=torch.float64)
    residual = pde_residual(lambda z: z[:, :1]**2 + z[:, 1:], coords, HeatCoefficients(s.exp(), 0.875))
    residual.sum().backward()
    assert s.grad.item() == pytest.approx(-2 * len(coords) * 0.10)


def test_inverse_checkpoint(tmp_path):
    config = {"model": {"width": 32, "hidden_layers": 3}}
    model = make_model(**config["model"])
    s = log_diffusivity(0.031)
    path = tmp_path / "model.pt"
    save_inverse_checkpoint(path, model, s, config)
    restored, restored_s, metadata = load_inverse_checkpoint(path)
    assert metadata == config and restored_s.requires_grad
    torch.testing.assert_close(restored_s, s, rtol=0, atol=0)
    coords = torch.rand((8, 2), dtype=torch.float64)
    torch.testing.assert_close(restored(coords), model(coords), rtol=0, atol=0)


def test_parameter_metrics_and_noiseless_least_squares():
    errors = parameter_metrics(0.024, 0.03)
    assert errors["signed_relative_error"] == pytest.approx(-0.2)
    assert errors["absolute_relative_error"] == pytest.approx(0.2)
    rod = Rod(0.030, 0.875, 0.60)
    coords, _, observed, _ = make_observations(rod.exact, 100, 0.0, 67507, 0.0)
    fit = least_squares_reference(coords, observed, rod.loss_rate, rod.amplitude, 0.10)
    assert fit.success and np.exp(fit.x[0]) == pytest.approx(rod.kappa, abs=1e-10)


def test_lbfgs_trials_are_distinct_from_iterations_and_updates():
    x = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
    optimizer = MonitoredLBFGS([x], lr=10, max_iter=1, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = x.square()
        loss.backward()
        return loss

    optimizer.step(closure)
    counts = optimizer.diagnostics(x.square().item(), 2 * x.detach())
    assert counts["lbfgs_iterations"] == counts["lbfgs_applied_updates"] == 1
    assert counts["lbfgs_closure_evaluations"] > 1
    assert counts["lbfgs_stop_reason"] == "iteration_limit"
