"""Train and evaluate one forward PINN from a JSON configuration."""

import argparse
import csv
import json
import os
import platform
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import numpy as np
import torch

from heat import Rod
from lbfgs_diagnostics import MonitoredLBFGS
from pinn import field_metrics, load_checkpoint, loss_terms, make_model, sample_points, save_checkpoint
from plot_forward import plot_results


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def train(config, rod, out):
    start = perf_counter()
    torch.manual_seed(config["seed"])
    generator = torch.Generator().manual_seed(config["seed"])
    model = make_model(**config["model"])
    points = sample_points(config["points"], generator)
    initial_target = torch.from_numpy(rod.initial(points[2][:, :1].numpy()))
    return optimize(config, out, model, lambda: loss_terms(model, points, initial_target, rod), start)


def optimize(config, out, model, objective, start, log_diffusivity=None):
    adam_settings = config["adam"]
    groups = [{"params": list(model.parameters()), "lr": adam_settings["lr_start"]}]
    if log_diffusivity is not None:
        groups.append({"params": [log_diffusivity], "lr": adam_settings["diffusivity_lr"]})
    parameters = [p for group in groups for p in group["params"]]
    adam = torch.optim.Adam(groups, lr=adam_settings["lr_start"])
    lbfgs = MonitoredLBFGS(parameters, **config["lbfgs"])
    write_json(out / "config.json", {
        **config,
        "sampling": "Fixed uniform full batches with an even split between boundaries",
        "initialization": "PyTorch Linear default, separate generator for sampling",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "optimizer_defaults": {"adam": adam.defaults, "lbfgs": lbfgs.defaults},
        "software": {"python": platform.python_version(), "torch": str(torch.__version__),
                     "numpy": np.__version__, "platform": platform.platform(),
                     "cpu": platform.processor()},
    })
    history = []
    weights = config["loss_weights"]

    def evaluate(phase, index):
        terms = objective()
        total = sum(weights[name] * value for name, value in terms.items())
        row = dict(phase=phase, evaluation=index, total=total.detach().item(),
                   **{name: value.detach().item() for name, value in terms.items()})
        if log_diffusivity is not None:
            row["kappa"] = log_diffusivity.detach().exp().item()
        if not np.isfinite(row["total"]):
            if history:
                write_csv(out / "partial_loss_history.csv", history)
            raise FloatingPointError(f"Nonfinite loss at {phase} evaluation {index}")
        history.append(row)
        return total, row

    steps = adam_settings["steps"]
    lr_ratio = adam_settings["lr_end"] / adam_settings["lr_start"]
    for step in range(1, steps + 1):
        adam.param_groups[0]["lr"] = adam_settings["lr_start"] * lr_ratio ** (
            (step - 1) / (steps - 1)
        )
        adam.zero_grad(set_to_none=True)
        loss, row = evaluate("adam", step)
        loss.backward()
        adam.step()
        if step % 1000 == 0:
            print(f"Adam {step}/{steps} | loss {row['total']:.4e}", flush=True)

    _, after_adam = evaluate("after_adam", 1)
    print(f"After Adam | loss {after_adam['total']:.4e}", flush=True)
    closures = 0

    def closure():
        nonlocal closures
        closures += 1
        lbfgs.zero_grad(set_to_none=True)
        loss, row = evaluate("lbfgs_closure", closures)
        loss.backward()
        if closures % 250 == 0:
            print(f"L-BFGS closure {closures} | trial loss {row['total']:.4e}", flush=True)
        return loss

    lbfgs.step(closure)
    runtime = perf_counter() - start
    loss, final = evaluate("final", 1)
    gradient = torch.cat([g.flatten() for g in torch.autograd.grad(loss, parameters)])
    counts = {"adam_step_calls": steps, "adam_closure_evaluations": 0,
              **lbfgs.diagnostics(final["total"], gradient)}
    assert counts["lbfgs_closure_evaluations"] == closures
    return model.eval(), history, after_adam, final, counts, runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "pinn_config.json")
    parser.add_argument("--seed", type=int, help="Override the configuration seed")
    parser.add_argument("--output", type=Path, help="New directory for the run artifacts")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.seed is not None:
        config["seed"] = args.seed
    rod = Rod(**config["rod"])
    if (config["device"], config["dtype"]) != ("cpu", "float64"):
        raise ValueError("The forward experiment supports CPU float64.")
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    reference = ROOT / config["reference"]
    with np.load(reference) as data:
        x, t, exact, cn = (data[name] for name in ("x", "t", "exact", "cn"))
    nodes = np.linspace(0, 1, 101)
    np.testing.assert_allclose(x, nodes)
    np.testing.assert_allclose(t, nodes)
    np.testing.assert_allclose(exact, rod.exact(x, t[:, None]), rtol=0, atol=1e-14)
    summary = json.loads(reference.with_name("benchmark.json").read_text(encoding="utf-8"))
    if summary["case"]["rod"] != config["rod"]:
        raise ValueError("Reference Crank–Nicolson coefficients differ from the training case.")

    out = args.output or ROOT / "results" / "forward" / f"seed-{config['seed']}"
    out.mkdir(parents=True, exist_ok=False)
    try:
        model, history, after_adam, final, counts, runtime = train(config, rod, out)
    except Exception as error:
        write_json(out / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    save_checkpoint(out / "model.pt", model, config)
    write_csv(out / "loss_history.csv", history)

    xx, tt = np.meshgrid(x, t)
    coords = torch.from_numpy(np.column_stack((xx.ravel(), tt.ravel())))
    restored, restored_config = load_checkpoint(out / "model.pt")
    assert restored_config == config
    with torch.inference_mode():
        values = model(coords)
        torch.testing.assert_close(restored(coords), values, rtol=0, atol=0)
        prediction = values.numpy().reshape(exact.shape)
    pinn_metrics, pinn_time = field_metrics(prediction, exact)
    cn_metrics, cn_time = field_metrics(cn, exact)
    np.savez_compressed(out / "predictions.npz", x=x, t=t, pinn=prediction, exact=exact, cn=cn)
    write_csv(out / "error_by_time.csv", [
        dict(t=time, pinn_relative_l2=p, cn_relative_l2=c)
        for time, p, c in zip(t, pinn_time, cn_time)
    ])
    metrics = {
        "seed": config["seed"], "pinn": pinn_metrics, "crank_nicolson": cn_metrics,
        "after_adam": after_adam, "final_loss": final, "optimizer_counts": counts,
        "training_seconds": runtime, "crank_nicolson_mesh": summary["finest_mesh"],
        "crank_nicolson_median_seconds": float(np.median(
            summary["timed_solve_seconds"][str(summary["finest_mesh"]["nx"])]
        )),
    }
    plot_results(out, x, t, prediction, exact, cn, pinn_time, cn_time, history)
    write_json(out / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
