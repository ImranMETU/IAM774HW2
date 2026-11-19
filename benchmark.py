"""Run the deterministic Phase 1 benchmark and regenerate its data and figures."""

import argparse
import csv
import json
import os
import platform
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from heat import Rod, crank_nicolson, relative_l2


def plot_profiles(out, x, t, exact, cn):
    fig, (profiles, error) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for fraction, color in zip((0, 0.1, 0.3, 1), ("#324d79", "#007d76", "#ad5233", "#77467f")):
        i = round(fraction * (len(t) - 1))
        profiles.plot(x, exact[i], color=color, label=f"t = {t[i]:g}")
        profiles.plot(x[::5], cn[i, ::5], "o", ms=4, mfc="none", color=color)
    profiles.set(title="Exact profiles (lines) and CN samples (circles)",
                 xlabel="Position x", ylabel="Temperature excess u")
    profiles.legend(frameon=False, ncols=2)
    absolute = np.abs(cn - exact)
    mesh = error.pcolormesh(x, t, absolute, shading="nearest", cmap="viridis")
    error.set(title="Absolute error on the reference grid", xlabel="Position x", ylabel="Time t",
              xlim=(0, 1), ylim=(0, 1))
    fig.colorbar(mesh, ax=error, label=r"$|u_{\mathrm{CN}} - u_{\mathrm{exact}}|$", format="%.1e")
    fig.savefig(out / "profiles.png", dpi=160)
    plt.close(fig)


def plot_convergence(out, rows):
    h = np.array([1 / r["nx"] for r in rows])
    err = np.array([r["relative_l2_t1"] for r in rows])
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    ax.loglog(h, err, "o-", color="#007d76", lw=2, label="Measured final-time error")
    ax.loglog(h, 2 * err[-1] * (h / h[-1])**2, "--", color="#4c5063", label="Second-order slope")
    ax.set(title="Crank–Nicolson mesh refinement", xlabel="Spatial step Δx",
           ylabel="Relative L² error at t = 1")
    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.18)
    fig.savefig(out / "convergence.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    case = json.loads((ROOT / "case.json").read_text())
    rod = Rod(**case["rod"])
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    rows, timings = [], {}
    for nx in case["space_intervals"]:
        nt = nx * case["time_steps_per_space_interval"]
        crank_nicolson(rod, nx, nt)  # warm up this mesh before timing
        elapsed = []
        for _ in range(case["timed_repeats"]):
            start = perf_counter()
            solution = crank_nicolson(rod, nx, nt)
            elapsed.append(perf_counter() - start)
            x, t, field = solution
        error = relative_l2(field[-1], rod.exact(x, t[-1]))
        order = None
        if rows:
            order = np.log(rows[-1]["relative_l2_t1"] / error) / np.log(nx / rows[-1]["nx"])
        rows.append(dict(nx=nx, nt=nt, relative_l2_t1=error, order=order,
                         median_seconds=median(elapsed)))
        timings[nx] = elapsed

    with (out / "convergence.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Restrict the last, finest mesh to common nodes. No interpolation error.
    n = case["evaluation_intervals"]
    if nx % n or nt % n:
        raise ValueError("The finest mesh must contain every evaluation node.")
    x_eval, t_eval = x[::nx // n], t[::nt // n]
    cn = field[::nt // n, ::nx // n]
    exact = rod.exact(x_eval, t_eval[:, None])
    np.savez_compressed(out / "reference_grid.npz", x=x_eval, t=t_eval, exact=exact, cn=cn)
    plot_profiles(out, x_eval, t_eval, exact, cn)
    plot_convergence(out, rows)
    summary = {
        "case": case, "modal_rates": {m: rod.decay_rate(m) for m in (1, 4)},
        "reference_grid_relative_l2": relative_l2(cn, exact),
        "finest_mesh": {"nx": nx, "nt": nt}, "timed_solve_seconds": timings,
        "runtime": {"python": sys.version.split()[0], "platform": platform.platform(),
                    "cpu": platform.processor(), "numpy": np.__version__, "scipy": scipy.__version__,
                    "matplotlib": matplotlib.__version__, "blas_threads_requested": 1},
    }
    (out / "benchmark.json").write_text(json.dumps(summary, indent=2) + "\n")
    print((out / "convergence.csv").read_text().strip())
    print(f"Common-grid relative L2: {summary['reference_grid_relative_l2']:.8e}")


if __name__ == "__main__":
    main()
