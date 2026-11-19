"""Summarize the three forward runs and score a common held-out interior batch."""

import argparse
import json

import numpy as np
import torch

from heat import Rod
from pinn import load_checkpoint, pde_residual, sample_points
from train_forward import ROOT, write_csv, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout-seed", type=int, default=97507)
    parser.add_argument("--heldout-count", type=int, default=10000)
    args = parser.parse_args()
    root = ROOT / "results" / "forward"
    torch.set_num_threads(1)
    heldout = torch.rand((args.heldout_count, 2), dtype=torch.float64,
                         generator=torch.Generator().manual_seed(args.heldout_seed))
    if not torch.all((heldout > 0) & (heldout < 1)):
        raise ValueError("Held-out points must be strictly interior.")
    heldout_set = set(map(tuple, heldout.tolist()))
    runs, residuals = [], {}
    expected = json.loads((ROOT / "pinn_config.json").read_text())
    for seed in (67507, 67508, 67509):
        directory = root / f"seed-{seed}"
        model, config = load_checkpoint(directory / "model.pt")
        if config != {**expected, "seed": seed}:
            raise ValueError(f"Forward settings differ for seed {seed}.")
        train, _, _ = sample_points(config["points"], torch.Generator().manual_seed(seed))
        if not heldout_set.isdisjoint(map(tuple, train.tolist())):
            raise ValueError(f"Held-out points overlap the training batch for {seed}.")
        metrics = json.loads((directory / "metrics.json").read_text())
        counts = metrics["optimizer_counts"]
        diagnostics_source = "original run"
        if seed == 67507 and "lbfgs_stop_reason" not in counts:
            # The original run did not log the stop reason. Require an exact replay
            # before attributing replay diagnostics to the original result.
            replay_dir = root / "diagnostics-67507"
            replay = json.loads((replay_dir / "metrics.json").read_text())
            replay_counts = replay["optimizer_counts"]
            with np.load(directory / "predictions.npz") as a, np.load(replay_dir / "predictions.npz") as b:
                np.testing.assert_array_equal(a["pinn"], b["pinn"])
            assert replay["final_loss"] == metrics["final_loss"]
            assert all(replay_counts[key] == value for key, value in counts.items())
            counts = replay_counts
            diagnostics_source = "exact replay"
        residual = pde_residual(model, heldout, Rod(**config["rod"])).detach().numpy().ravel()
        residuals[f"residual_{seed}"] = residual
        rms = float(np.sqrt(np.mean(residual**2)))
        runs.append({
            "seed": seed, **metrics["pinn"], "training_seconds": metrics["training_seconds"],
            **{f"final_{k}": metrics["final_loss"][k] for k in ("pde", "bc", "ic")},
            "heldout_residual_rms": rms,
            "heldout_residual_max_abs": float(np.max(np.abs(residual))),
            **counts,
            "stopping_diagnostics_source": diagnostics_source,
        })
    fields = ("relative_l2", "relative_l2_t1", "max_absolute_error", "training_seconds",
              "final_pde", "final_bc", "final_ic", "heldout_residual_rms",
              "heldout_residual_max_abs")
    statistics = {}
    for field in fields:
        values = [row[field] for row in runs]
        statistics[field] = {"mean": float(np.mean(values)), "sample_std": float(np.std(values, ddof=1))}
    write_csv(root / "robustness.csv", runs)
    write_json(root / "robustness.json", {
        "heldout_seed": args.heldout_seed, "heldout_count": args.heldout_count,
        "runs": runs, "statistics": statistics,
    })
    np.savez_compressed(root / "heldout_residuals.npz", coordinates=heldout.numpy(), **residuals)
    print(json.dumps(statistics, indent=2))


if __name__ == "__main__":
    main()
