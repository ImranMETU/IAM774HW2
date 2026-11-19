import numpy as np
import pytest
from numpy.testing import assert_allclose

from heat import Rod, crank_nicolson, relative_l2


ROD = Rod(0.03, 0.875, 0.60)


def test_exact_initial_and_boundary_conditions():
    x = np.linspace(0, 1, 91)
    expected = np.sin(np.pi * x) + 0.60 * np.sin(4 * np.pi * x)
    assert_allclose(ROD.exact(x, 0), expected, atol=1e-15)
    t = np.linspace(0, 1, 53)[:, None]
    assert_allclose(ROD.exact([0, 1], t), 0, atol=1e-15)


def test_exact_satisfies_pde_away_from_boundaries():
    x = np.linspace(0.04, 0.96, 19)
    t = np.array([0.06, 0.39, 0.88])[:, None]
    h = 1e-4
    value = ROD.exact(x, t)
    ut = (ROD.exact(x, t + h) - ROD.exact(x, t - h)) / (2 * h)
    uxx = (ROD.exact(x + h, t) - 2 * value + ROD.exact(x - h, t)) / h**2
    assert_allclose(ut, ROD.kappa * uxx - ROD.loss_rate * value, atol=6e-7)


def test_cn_matches_discrete_mode_amplification():
    nx, nt = 32, 17
    x, t, field = crank_nicolson(ROD, nx, nt)
    expected = np.zeros_like(field)
    for mode, weight in [(1, 1), (4, ROD.amplitude)]:
        rate = 4 * ROD.kappa * nx**2 * np.sin(mode * np.pi / (2 * nx))**2
        rate += ROD.loss_rate
        gain = (1 - rate / (2 * nt)) / (1 + rate / (2 * nt))
        expected += weight * gain ** np.arange(len(t))[:, None] * np.sin(mode * np.pi * x)
    assert_allclose(field, expected, atol=5e-14)
    assert np.all(field[:, [0, -1]] == 0)


def test_cn_second_order_at_final_time():
    errors = []
    for nx in [50, 100, 200, 400]:
        x, t, field = crank_nicolson(ROD, nx, 2 * nx)
        errors.append(relative_l2(field[-1], ROD.exact(x, t[-1])))
    orders = np.log2(np.array(errors[:-1]) / errors[1:])
    assert np.all((1.95 < orders) & (orders < 2.05)), orders


def test_temporal_order_against_semidiscrete_solution():
    # Holding space fixed removes spatial truncation error from this test.
    rod = Rod(0.03, 0.875, 0.0)
    nx = 40
    rate = rod.loss_rate + 4 * rod.kappa * nx**2 * np.sin(np.pi / (2 * nx))**2
    errors = []
    for nt in [8, 16, 32]:
        x, _, field = crank_nicolson(rod, nx, nt)
        errors.append(relative_l2(field[-1], np.sin(np.pi * x) * np.exp(-rate)))
    assert_allclose(np.array(errors[:-1]) / errors[1:], 4, rtol=0.01)


def test_relative_l2_uses_full_field():
    ref = np.array([[3., 4.], [0., 0.]])
    assert relative_l2(ref + 1, ref) == pytest.approx(0.4)
