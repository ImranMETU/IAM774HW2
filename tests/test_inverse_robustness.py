import json

import numpy as np
import torch

from heat import Rod
from inverse_robustness import BASELINE, SEEDS, experiment_plan
from train_inverse import ROOT, observation_data
import train_inverse


def base_config():
    return json.loads((ROOT / "inverse_config.json").read_text())


def test_seed_pairings_and_collocation_control(tmp_path):
    base = base_config()
    plan = list(experiment_plan(base, tmp_path))
    assert len(plan) == 6
    for experiment, varied, config, directory, source in plan:
        network, observation = (varied, SEEDS[0]) if experiment == "initialization" else (SEEDS[0], varied)
        assert (config["seed"], config["observation_seed"]) == (network, observation)
        assert config["collocation_seed"] == SEEDS[0]
        assert config == {**base, "seed": network, "observation_seed": observation, "collocation_seed": SEEDS[0]}
        assert (directory == BASELINE) == (varied == SEEDS[0])
        assert source == (BASELINE / "observations.npz" if observation == SEEDS[0] else None)


def test_observation_replay_and_rng_isolation():
    base = base_config()
    truth = Rod(base["kappa_true"], base["loss_rate"], base["amplitude"])
    archived = observation_data(base, truth, BASELINE / "observations.npz")
    torch.manual_seed(934)
    state = torch.random.get_rng_state()
    for seed in SEEDS:
        config = {**base, "observation_seed": seed}
        draw = observation_data(config, truth)
        retry = observation_data(config, truth)
        for actual, expected in zip(draw, retry):
            np.testing.assert_array_equal(actual, expected)
        if seed == SEEDS[0]:
            for actual, expected in zip(draw, archived):
                np.testing.assert_array_equal(actual, expected)
        else:
            assert not np.array_equal(draw[0], archived[0])
            assert not np.array_equal(draw[2] - draw[1], archived[2] - archived[1])
    assert torch.equal(torch.random.get_rng_state(), state)


def test_training_initialization_changes_only_with_network_seed(monkeypatch, tmp_path):
    captures = []
    batches = []
    real_sample = train_inverse.sample_points

    def sample(counts, generator):
        points = real_sample(counts, generator)
        batches.append(points)
        return points

    def capture(config, out, model, objective, start, s):
        captures.append(torch.cat([p.detach().flatten() for p in model.parameters()]))
        return None

    monkeypatch.setattr(train_inverse, "optimize", capture)
    monkeypatch.setattr(train_inverse, "sample_points", sample)
    base = base_config()
    truth = Rod(base["kappa_true"], base["loss_rate"], base["amplitude"])
    for network, observation in [(67507, 67507), (67507, 67508), (67508, 67507)]:
        config = {**base, "seed": network, "observation_seed": observation, "collocation_seed": 67507}
        coords, _, noisy, _ = observation_data(config, truth)
        train_inverse.train(config, tmp_path, coords, noisy)
    assert torch.equal(captures[0], captures[1])
    assert not torch.equal(captures[0], captures[2])
    for points in batches[1:]:
        assert all(torch.equal(a, b) for a, b in zip(points, batches[0]))
