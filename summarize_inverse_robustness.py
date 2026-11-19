"""Validate and summarize saved inverse robustness runs."""

import argparse
import csv
import json
from pathlib import Path

from inverse_robustness import BASELINE, DEFAULT_OUTPUT, SEEDS, experiment_plan
from train_inverse import ROOT, observation_data, write_csv, write_json

import matplotlib.pyplot as plt
import numpy as np

from heat import Rod


PARAMETER_FIELDS = [
    f"{method}_{metric}" for method in ("pinn", "least_squares")
    for metric in ("kappa", "signed_relative_error", "absolute_relative_error")
]
NUMERICAL_FIELDS = PARAMETER_FIELDS + [
    "noise_sigma", "pinn_field_l2", "least_squares_field_l2",
    "pinn_observation_rmse", "least_squares_observation_rmse", "training_seconds",
    "pde_loss", "bc_loss", "ic_loss", "data_loss", "total_loss", "kappa_after_adam",
    "adam_kappa_min", "adam_kappa_max", "lbfgs_kappa_min", "lbfgs_kappa_max",
]
DIAGNOSTICS = ["adam_step_calls", "lbfgs_iterations", "lbfgs_applied_updates",
               "lbfgs_closure_evaluations", "lbfgs_stop_reason", "least_squares_success"]
FIELDS = ["experiment", "varied_seed", "network_seed", "observation_seed", "collocation_seed",
          "status", "failure", "artifacts"] + NUMERICAL_FIELDS + DIAGNOSTICS


def read_history(directory):
    with (directory / "diffusivity_history.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def phase_values(history, phase, endpoint):
    values = [float(row["kappa"]) for row in history if row["phase"] == phase]
    values.append(next(float(row["kappa"]) for row in history if row["phase"] == endpoint))
    return values


def collect(base, output):
    truth = Rod(base["kappa_true"], base["loss_rate"], base["amplitude"])
    rows, histories, datasets = [], {}, {}
    for experiment, seed, config, directory, source in experiment_plan(base, output):
        row = dict.fromkeys(FIELDS)
        resolved = directory.resolve()
        artifacts = (
            resolved.relative_to(ROOT).as_posix() if resolved.is_relative_to(ROOT)
            else resolved.relative_to(output.resolve()).as_posix()
        )
        row.update(experiment=experiment, varied_seed=seed, network_seed=config["seed"],
                   observation_seed=config["observation_seed"], collocation_seed=config["collocation_seed"],
                   artifacts=artifacts)
        rows.append(row)
        failure = directory / "failure.json"
        metrics_file = directory / "metrics.json"
        if failure.exists() or not metrics_file.exists():
            row["status"] = "failed" if failure.exists() else "incomplete"
            row["failure"] = json.loads(failure.read_text())["message"] if failure.exists() else None
            continue

        saved = json.loads((directory / "experiment.json").read_text())
        if directory == BASELINE:
            saved["collocation_seed"] = saved["seed"]
        if any(saved[key] != value for key, value in config.items()):
            raise ValueError(f"Configuration differs from experiment plan for {directory}.")
        archive = directory / "observations.npz"
        actual = observation_data(config, truth, archive)
        expected = observation_data(config, truth, source)
        for actual_value, expected_value in zip(actual, expected):
            np.testing.assert_array_equal(actual_value, expected_value)
        obs_seed = config["observation_seed"]
        for other_seed, (coordinates, noise) in datasets.items():
            if obs_seed != other_seed:
                assert not np.array_equal(actual[0], coordinates)
                assert not np.array_equal(actual[2] - actual[1], noise)
        datasets[obs_seed] = actual[0], actual[2] - actual[1]

        metrics = json.loads(metrics_file.read_text())
        least_squares = metrics["least_squares"]
        counts = metrics["optimizer_counts"]
        history = read_history(directory)
        adam = phase_values(history, "adam", "after_adam")
        lbfgs = phase_values(history, "lbfgs_closure", "final")
        assert len(adam) - 1 == counts["adam_step_calls"]
        assert len(lbfgs) - 1 == counts["lbfgs_closure_evaluations"]
        assert lbfgs[-1] == metrics["pinn"]["kappa"]
        assert metrics["noise_sigma"] == actual[-1]
        histories[experiment, seed] = adam, lbfgs
        row.update(
            status="complete" if least_squares["success"] else "least_squares_failed",
            failure=None if least_squares["success"] else least_squares["message"],
            noise_sigma=actual[-1], training_seconds=metrics["training_seconds"],
            kappa_after_adam=adam[-1], adam_kappa_min=min(adam), adam_kappa_max=max(adam),
            lbfgs_kappa_min=min(lbfgs), lbfgs_kappa_max=max(lbfgs),
            least_squares_success=least_squares["success"],
        )
        row.update({key: counts[key] for key in DIAGNOSTICS if key in counts})
        row.update({f"{key}_loss": metrics["final_loss"][key] for key in ("pde", "bc", "ic", "data", "total")})
        for method in ("pinn", "least_squares"):
            result = metrics[method]
            for field in ("kappa", "signed_relative_error", "absolute_relative_error", "observation_rmse"):
                row[f"{method}_{field}"] = result[field]
            row[f"{method}_field_l2"] = result["field"]["relative_l2"]
    return rows, histories


def statistics(rows):
    summary = {}
    for experiment in ("initialization", "observation_noise"):
        group = [row for row in rows if row["experiment"] == experiment]
        stats = {}
        for field in NUMERICAL_FIELDS:
            values = [
                row[field] for row in group if row[field] is not None
                and (not field.startswith("least_squares_") or row["least_squares_success"])
            ]
            stats[field] = {
                "count": len(values),
                "mean": float(np.mean(values)) if values else None,
                "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            }
        summary[experiment] = stats
    return summary


def plot_summary(output, rows, histories, kappa_true):
    groups = [("initialization", "Initialization seed"),
              ("observation_noise", "Observation dataset seed")]
    colors = ["#344e80", "#007d76", "#b95435"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for (experiment, label), panels in zip(groups, axes):
        for ax, index, phase, xlabel in zip(
            panels, (0, 1), ("Adam", "L-BFGS"),
            ("Adam update (pre-update)", "Closure evaluation (includes trial points)"),
        ):
            ax.axhline(kappa_true, ls="--", lw=1, color="#575b65", label="True diffusivity")
            for seed, color in zip(SEEDS, colors):
                if (experiment, seed) not in histories:
                    continue
                values = histories[experiment, seed][index]
                ax.plot(range(1, len(values) + 1), values, color=color, label=str(seed))
                ax.plot(len(values), values[-1], "o", ms=4, color=color)
            ax.set(title=f"{label} · {phase}", xlabel=xlabel, ylabel="Diffusivity κ")
            ax.grid(alpha=0.2)
        panels[0].legend(ncols=2, frameon=False)
    fig.savefig(output / "kappa_convergence.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(9, 4), layout="constrained")
    for ax, (experiment, label) in zip(axes, groups):
        group = [row for row in rows if row["experiment"] == experiment]
        for method, name, offset, marker, color in (
            ("pinn", "PINN", -0.12, "o", "#344e80"),
            ("least_squares", "Least squares", 0.12, "s", "#b95435"),
        ):
            points = [(i + offset, 100 * row[f"{method}_signed_relative_error"])
                      for i, row in enumerate(group)
                      if row[f"{method}_signed_relative_error"] is not None
                      and (method == "pinn" or row["least_squares_success"])]
            if points:
                ax.scatter(*zip(*points), marker=marker, color=color, label=name, s=42)
        ax.axhline(0, color="#575b65", lw=1, ls="--")
        ax.set_xticks(range(len(group)), [str(row["varied_seed"]) for row in group])
        ax.set_xlabel(label)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Signed relative diffusivity error (%)")
    axes[0].legend(frameon=False)
    fig.savefig(output / "parameter_error_comparison.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    base = json.loads((ROOT / "inverse_config.json").read_text())
    rows, histories = collect(base, args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "inverse_robustness_summary.csv", rows)
    keys = FIELDS[:6] + PARAMETER_FIELDS
    write_csv(args.output / "parameter_estimates_by_seed.csv", [{key: row[key] for key in keys} for row in rows])
    write_json(args.output / "summary_statistics.json", statistics(rows))
    plot_summary(args.output, rows, histories, base["kappa_true"])
    for row in rows:
        print(row["experiment"], row["varied_seed"], row["status"], row["pinn_kappa"], row["least_squares_kappa"])


if __name__ == "__main__":
    main()
