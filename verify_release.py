"""Recompute archived metrics, reload checkpoints, and regenerate every result plot."""

import argparse
import csv
import json
from pathlib import Path

from benchmark import ROOT, plot_convergence, plot_profiles
from inverse import load_inverse_checkpoint
from pinn import field_metrics, load_checkpoint
from plot_forward import plot_results
from summarize_inverse_robustness import DEFAULT_OUTPUT, collect, plot_summary, statistics
from train_forward import write_json
from train_inverse import plot_inverse

import matplotlib.image as image
import numpy as np
import torch


def records(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_metric_set(actual, expected):
    for key, value in actual.items():
        np.testing.assert_allclose(value, expected[key], rtol=1e-10, atol=1e-14, err_msg=key)


def check_run(run, out, reference, inverse):
    target = out / run.relative_to(ROOT / "results")
    target.mkdir(parents=True)
    stored = read_json(run / "metrics.json")
    history = records(run / "loss_history.csv")
    for row in history:
        for field in row.keys() - {"phase"}:
            row[field] = float(row[field])
    losses = ("pde", "bc", "ic", "data") if inverse else ("pde", "bc", "ic")
    checkpoint_loader = load_inverse_checkpoint if inverse else load_checkpoint
    model, *restored = checkpoint_loader(run / "model.pt")
    config = restored[-1]
    for row in history:
        total = sum(config["loss_weights"][field] * row[field] for field in losses)
        np.testing.assert_allclose(total, row["total"], rtol=1e-13, atol=0)
    for phase, key in (("after_adam", "after_adam"), ("final", "final_loss")):
        row = next(row for row in history if row["phase"] == phase)
        check_metric_set({field: row[field] for field in ("total", *losses)}, stored[key])
    counts = stored["optimizer_counts"]
    assert sum(row["phase"] == "adam" for row in history) == counts["adam_step_calls"]
    assert sum(row["phase"] == "lbfgs_closure" for row in history) == counts["lbfgs_closure_evaluations"]

    with np.load(run / "predictions.npz") as data:
        fields = dict(data)
    for key in ("x", "t", "exact"):
        np.testing.assert_array_equal(fields[key], reference[key])
    x, t, pinn, exact = (fields[key] for key in ("x", "t", "pinn", "exact"))
    xx, tt = np.meshgrid(x, t)
    coords = torch.from_numpy(np.column_stack((xx.ravel(), tt.ravel())))
    with torch.inference_mode():
        predicted = model(coords).numpy().reshape(pinn.shape)
    np.testing.assert_allclose(predicted, pinn, rtol=0, atol=1e-12)
    replay_delta = float(np.max(np.abs(predicted - pinn)))
    metrics, pinn_time = field_metrics(pinn, exact)
    expected = stored["pinn"]["field"] if inverse else stored["pinn"]
    check_metric_set(metrics, expected)
    time_rows = records(run / "error_by_time.csv")
    np.testing.assert_array_equal([float(row["t"]) for row in time_rows], t)
    np.testing.assert_allclose([float(row["pinn_relative_l2"]) for row in time_rows], pinn_time)
    method = "least_squares" if inverse else "cn"
    comparison, comparison_time = field_metrics(fields[method], exact)
    expected = stored["least_squares"]["field"] if inverse else stored["crank_nicolson"]
    check_metric_set(comparison, expected)
    np.testing.assert_allclose([float(row[f"{method}_relative_l2"]) for row in time_rows], comparison_time)
    if inverse:
        s = restored[0]
        np.testing.assert_allclose(s.exp().item(), stored["pinn"]["kappa"], rtol=1e-14)
        table = records(run / "observed_vs_predicted.csv")
        with np.load(run / "observations.npz") as archive:
            observed = archive["observed"]
            observation_coords = archive["coordinates"]
        np.testing.assert_array_equal([[float(r["x"]), float(r["t"])] for r in table], observation_coords)
        np.testing.assert_array_equal([float(r["observed"]) for r in table], observed)
        pinn_observed = np.array([float(row["pinn"]) for row in table])
        ls_observed = np.array([float(row["least_squares"]) for row in table])
        with torch.inference_mode():
            replay = model(torch.from_numpy(observation_coords)).numpy().ravel()
        np.testing.assert_allclose(replay, pinn_observed, rtol=0, atol=1e-12)
        for key, values in (("pinn", pinn_observed), ("least_squares", ls_observed)):
            check_metric_set({"observation_rmse": np.sqrt(np.mean((values - observed)**2))}, stored[key])
        truth = read_json(run / "experiment.json")["kappa_true"]
        plot_inverse(target, history, observed, pinn_observed, ls_observed, truth)
    else:
        np.testing.assert_array_equal(fields["cn"], reference["cn"])
        plot_results(target, x, t, pinn, exact, fields["cn"], pinn_time, comparison_time, history)
    return replay_delta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".reproduction" / "figure-check")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    with np.load(ROOT / "results" / "reference_grid.npz") as data:
        reference = dict(data)
    rows = [{key: float(row[key]) for key in ("nx", "relative_l2_t1")}
            for row in records(ROOT / "results" / "convergence.csv")]
    plot_convergence(out, rows)
    plot_profiles(out, *(reference[key] for key in ("x", "t", "exact", "cn")))
    predictions = {}
    for mode in ("forward", "inverse"):
        for path in sorted((ROOT / "results" / mode).rglob("model.pt")):
            run = path.parent
            delta = check_run(run, out, reference, inverse=mode == "inverse")
            predictions[run.relative_to(ROOT).as_posix()] = delta
    base = read_json(ROOT / "inverse_config.json")
    summary, histories = collect(base, DEFAULT_OUTPUT)
    assert statistics(summary) == read_json(DEFAULT_OUTPUT / "summary_statistics.json")
    target = out / DEFAULT_OUTPUT.relative_to(ROOT / "results")
    plot_summary(target, summary, histories, base["kappa_true"])
    originals = {p.relative_to(ROOT / "results") for p in (ROOT / "results").rglob("*.png")}
    copies = {p.relative_to(out) for p in out.rglob("*.png")}
    assert originals == copies, "Archived and regenerated figure paths differ."
    differences = [
        p.as_posix() for p in sorted(originals)
        if not np.array_equal(image.imread(ROOT / "results" / p), image.imread(out / p))
    ]
    report = {"checkpoints": len(predictions), "max_abs_checkpoint_delta_by_run": predictions,
              "figures": len(originals), "figures_with_pixel_differences": differences}
    write_json(out / "verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
