import numpy as np
import pytest
import torch

from heat import Rod
from pinn import (
    field_metrics, load_checkpoint, make_model, pde_residual,
    sample_points, save_checkpoint,
)


def test_model_shape_and_precision():
    model = make_model(32, 3)
    assert model(torch.zeros(12, 2, dtype=torch.float64)).shape == (12, 1)
    assert all(p.dtype == torch.float64 for p in model.parameters())


def test_residual_value_and_parameter_gradient():
    rod = Rod(0.030, 0.875, 0.60)
    scale = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    points = torch.tensor([[0.15, 0.3], [0.7, 0.2]], dtype=torch.float64)

    def field(z):
        return scale * z[:, :1].square() * z[:, 1:].exp()

    x, t = points[:, :1], points[:, 1:]
    expected = t.exp() * ((1 + rod.loss_rate) * x.square() - 2 * rod.kappa)
    result = pde_residual(field, points, rod)
    torch.testing.assert_close(result, scale * expected)
    result.sum().backward()
    torch.testing.assert_close(scale.grad, expected.sum())


def test_points_and_seed():
    counts = {"interior": 1000, "boundary": 100, "initial": 200}
    batch = sample_points(counts, torch.Generator().manual_seed(67507))
    repeated = sample_points(counts, torch.Generator().manual_seed(67507))
    for points, repeat, count in zip(batch, repeated, counts.values()):
        assert points.shape == (count, 2)
        assert torch.all((points >= 0) & (points <= 1))
        torch.testing.assert_close(points, repeat, rtol=0, atol=0)
    interior, boundary, initial = batch
    assert torch.all((interior > 0) & (interior < 1))
    assert int((boundary[:, 0] == 0).sum()) == 50
    assert int((boundary[:, 0] == 1).sum()) == 50
    assert torch.all(initial[:, 1] == 0)


def test_field_metrics():
    exact = np.array([[1., 2.], [2., 4.]])
    pred = np.array([[2., 4.], [2., 3.]])
    metrics, by_time = field_metrics(pred, exact)
    assert metrics["relative_l2"] == pytest.approx(np.sqrt(6 / 25))
    assert metrics["relative_l2_t1"] == pytest.approx(1 / np.sqrt(20))
    assert metrics["max_absolute_error"] == 2
    np.testing.assert_allclose(by_time, [1, 1 / np.sqrt(20)])


def test_checkpoint_roundtrip(tmp_path):
    config = {"model": {"width": 32, "hidden_layers": 3}, "seed": 67507}
    model = make_model(**config["model"])
    points = torch.rand(7, 2, dtype=torch.float64)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, config)
    restored, loaded = load_checkpoint(path)
    assert loaded == config
    torch.testing.assert_close(model(points), restored(points), rtol=0, atol=0)
