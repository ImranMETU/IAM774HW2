"""Fit one soft PINN and a least-squares reference to the same observations."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from train_forward import ROOT, optimize, write_csv, write_json

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch

from heat import Rod
from inverse import (
    inverse_terms, least_squares_reference, load_inverse_checkpoint,
    log_diffusivity, make_observations, parameter_metrics, save_inverse_checkpoint,
)
from pinn import field_metrics, make_model, sample_points


def train(config, out, coords, observations):
    start = perf_counter()
    torch.manual_seed(config["seed"])
    model = make_model(**config["model"])
    s = log_diffusivity(config["kappa_initial"])
    generator = torch.Generator().manual_seed(config.get("collocation_seed", config["seed"]))
    interior, boundary, initial = sample_points(config["points"], generator)
    coords, observations = torch.from_numpy(coords), torch.from_numpy(observations[:, None])
    points = (torch.cat((interior, coords)), boundary, initial)
    x = initial[:, :1]
    target = torch.sin(torch.pi * x) + config["amplitude"] * torch.sin(4 * torch.pi * x)

    def objective():
        return inverse_terms(model, s, config["loss_rate"], points, target, coords, observations)

    # Pass only optimizer settings and known quantities to the training routine.
    run = optimize(config, out, model, objective, start, s)
    return s, run


def plot_inverse(out, history, observed, pinn, classical, kappa_true):
    for filename, keys, ylabel in (
        ("loss_history.png", ("total", "pde", "bc", "ic", "data"), "Mean squared loss"),
        ("diffusivity_history.png", ("kappa",), "Diffusivity"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, layout="constrained")
        stages = [("adam", "after_adam", "Adam", "Update (before step)"),
                  ("lbfgs_closure", "final", "L-BFGS", "Closure evaluation (includes trials)")]
        for ax, (stage, end, title, xlabel) in zip(axes, stages):
            rows = [r for r in history if r["phase"] == stage]
            endpoint = next(r for r in history if r["phase"] == end)
            for key in keys:
                plot = ax.plot if key == "kappa" else ax.semilogy
                values = [r[key] for r in rows] + [endpoint[key]]
                line, = plot(range(1, len(values) + 1), values, label=key)
                ax.plot(len(values), values[-1], "o", color=line.get_color(), ms=4)
            if "kappa" in keys:
                ax.axhline(kappa_true, color="#72512e", ls="--", label="True diffusivity")
            ax.set(title=title, xlabel=xlabel)
            ax.grid(alpha=0.2)
        axes[0].set_ylabel(ylabel)
        axes[0].legend(frameon=False)
        fig.savefig(out / filename, dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True, layout="constrained")
    limits = (min(observed.min(), pinn.min(), classical.min()), max(observed.max(), pinn.max(), classical.max()))
    for ax, prediction, title in zip(axes, (pinn, classical), ("Inverse PINN", "Nonlinear least squares")):
        ax.plot(limits, limits, "--", lw=1, color="#444c59")
        ax.scatter(observed, prediction, s=15, alpha=0.7, color="#007d76")
        ax.set(title=title, xlabel="Observed temperature", aspect="equal")
    axes[0].set_ylabel("Predicted temperature")
    fig.savefig(out / "predicted_vs_observed.png", dpi=160)
    plt.close(fig)


def observation_data(config, truth, observation_file=None):
    if observation_file is None:
        return make_observations(
            truth.exact, config["observations"], config["eta"], config["observation_seed"], config["t_min"]
        )
    with np.load(observation_file) as archive:
        coords, clean, noisy, sigma = (archive[key] for key in ("coordinates", "clean", "observed", "sigma"))
    count = config["observations"]
    if coords.shape != (count, 2) or clean.shape != (count,) or noisy.shape != (count,):
        raise ValueError("Archived observations do not match the configured sample count.")
    np.testing.assert_allclose(clean, truth.exact(coords[:, 0], coords[:, 1]), rtol=0, atol=1e-14)
    np.testing.assert_allclose(sigma, config["eta"] * np.std(clean), rtol=1e-14, atol=0)
    return coords, clean, noisy, float(sigma)


def run_experiment(config, out, observation_file=None):
    if (config["device"], config["dtype"]) != ("cpu", "float64"):
        raise ValueError("The inverse experiment supports CPU float64.")
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    truth = Rod(config["kappa_true"], config["loss_rate"], config["amplitude"])
    with np.load(ROOT / config["reference"]) as reference:
        x, t, exact = (reference[key] for key in ("x", "t", "exact"))
    np.testing.assert_allclose(exact, truth.exact(x, t[:, None]), rtol=0, atol=1e-14)
    coords, clean, noisy, sigma = observation_data(config, truth, observation_file)
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "observations.npz", coordinates=coords, clean=clean, observed=noisy, sigma=sigma)
    known = {key: config[key] for key in (
        "seed", "model", "loss_rate", "amplitude", "kappa_initial", "points",
        "loss_weights", "adam", "lbfgs", "dtype", "device", "cpu_threads",
    )}
    if "collocation_seed" in config:
        known["collocation_seed"] = config["collocation_seed"]
    write_json(out / "experiment.json", {
        **config, "noise_sigma": sigma, "noise_std_ddof": 0, "observation_rng": "NumPy default_rng",
        "pde_batch": "Observation coordinates plus independent uniform interior points",
        "scipy": scipy.__version__,
        "observation_source": (
            observation_file.resolve().relative_to(ROOT).as_posix()
            if observation_file is not None and observation_file.resolve().is_relative_to(ROOT)
            else f"external archive ({observation_file.name})" if observation_file is not None else "generated"
        ),
    })
    try:
        s, run = train(known, out, coords, noisy)
    except Exception as error:
        write_json(out / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    model, history, after_adam, final, counts, elapsed = run
    save_inverse_checkpoint(out / "model.pt", model, s, known)
    restored, restored_s, restored_config = load_inverse_checkpoint(out / "model.pt")
    assert restored_config == known
    torch.testing.assert_close(restored_s, s, rtol=0, atol=0)
    with torch.inference_mode():
        xx, tt = np.meshgrid(x, t)
        grid = torch.from_numpy(np.column_stack((xx.ravel(), tt.ravel())))
        field = model(grid)
        torch.testing.assert_close(restored(grid), field, rtol=0, atol=0)
        prediction = field.numpy().reshape(exact.shape)
        observed_prediction = model(torch.from_numpy(coords)).numpy().ravel()

    fit = least_squares_reference(coords, noisy, truth.loss_rate, truth.amplitude, config["kappa_initial"])
    fitted_rod = Rod(float(np.exp(fit.x[0])), truth.loss_rate, truth.amplitude)
    classical_observations = fitted_rod.exact(coords[:, 0], coords[:, 1])
    classical_field = fitted_rod.exact(x, t[:, None])
    np.savez_compressed(out / "predictions.npz", x=x, t=t, pinn=prediction,
                        exact=exact, least_squares=classical_field)
    write_csv(out / "observed_vs_predicted.csv", [
        dict(x=x_i, t=t_i, observed=y, pinn=y_pinn, least_squares=y_ls)
        for (x_i, t_i), y, y_pinn, y_ls in zip(coords, noisy, observed_prediction, classical_observations)
    ])
    write_csv(out / "loss_history.csv", history)
    write_csv(out / "diffusivity_history.csv", [
        {key: row[key] for key in ("phase", "evaluation", "kappa")} for row in history
    ])
    pinn_field, pinn_by_time = field_metrics(prediction, exact)
    ls_field, ls_by_time = field_metrics(classical_field, exact)
    write_csv(out / "error_by_time.csv", [
        dict(t=time, pinn_relative_l2=p, least_squares_relative_l2=ls)
        for time, p, ls in zip(t, pinn_by_time, ls_by_time)
    ])
    metrics = {
        "seed": config["seed"], "noise_sigma": sigma,
        "pinn": {**parameter_metrics(final["kappa"], truth.kappa), "field": pinn_field,
                 "observation_rmse": float(np.sqrt(np.mean((observed_prediction - noisy)**2)))},
        "least_squares": {
            **parameter_metrics(fitted_rod.kappa, truth.kappa), "field": ls_field,
            "observation_rmse": float(np.sqrt(np.mean(fit.fun**2))),
            "success": bool(fit.success), "message": fit.message, "function_evaluations": fit.nfev,
        },
        "after_adam": after_adam, "final_loss": final, "optimizer_counts": counts,
        "training_seconds": elapsed,
    }
    write_json(out / "metrics.json", metrics)
    plot_inverse(out, history, noisy, observed_prediction, classical_observations, truth.kappa)
    print(json.dumps(metrics, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "inverse_config.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    out = args.output or ROOT / "results" / "inverse" / f"seed-{config['seed']}"
    run_experiment(config, out)


if __name__ == "__main__":
    main()
