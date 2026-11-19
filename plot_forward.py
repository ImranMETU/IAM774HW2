"""Plots for the forward PINN comparison."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_results(out, x, t, pinn, exact, cn, pinn_time, cn_time, history):
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True,
                             figsize=(9, 6), layout="constrained")
    for ax, time in zip(axes.flat, (0, 0.1, 0.3, 1)):
        i = np.argmin(np.abs(t - time))
        ax.plot(x, exact[i], color="#26344f", label="Analytical")
        ax.plot(x[::5], cn[i, ::5], "o", mfc="none", color="#008377", label="Crank–Nicolson")
        ax.plot(x, pinn[i], "--", color="#bb4a3a", label="PINN")
        ax.set_title(f"t = {t[i]:g}")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=3, frameon=False)
    fig.supxlabel("Position x")
    fig.supylabel("Temperature excess u")
    fig.savefig(out / "profiles.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    image = ax.pcolormesh(x, t, np.abs(pinn - exact), shading="nearest", cmap="magma")
    ax.set(title="PINN absolute error", xlabel="Position x", ylabel="Time t",
           xlim=(0, 1), ylim=(0, 1))
    fig.colorbar(image, ax=ax, label=r"$|u_\mathrm{PINN} - u_\mathrm{exact}|$")
    fig.savefig(out / "absolute_error.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(10, 4), layout="constrained")
    panels = [
        ("adam", "after_adam", "Adam", "Update number (pre-update loss)"),
        ("lbfgs_closure", "final", "L-BFGS", "Closure evaluation (including trial points)"),
    ]
    for ax, (phase, endpoint, title, xlabel) in zip(axes, panels):
        rows = [row for row in history if row["phase"] == phase]
        last = next(row for row in history if row["phase"] == endpoint)
        for name, label in [("total", "Total"), ("pde", "PDE"), ("bc", "BC"), ("ic", "IC")]:
            line, = ax.semilogy(np.arange(1, len(rows) + 2),
                                [row[name] for row in rows] + [last[name]], label=label)
            ax.plot(len(rows) + 1, last[name], "o", color=line.get_color(), ms=4)
        ax.set(title=title, xlabel=xlabel)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Mean squared loss")
    axes[0].legend(frameon=False)
    fig.savefig(out / "loss_history.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.semilogy(t, pinn_time, color="#bb4a3a", label="PINN")
    # At t=0 the numerical method uses the prescribed initial condition exactly
    # up to floating point. Exclude this point from the logarithmic plot.
    ax.semilogy(t[1:], cn_time[1:], color="#008377", label="Crank–Nicolson (t > 0)")
    ax.set(title="Error by time slice", xlabel="Time t", ylabel="Relative spatial L² error")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.savefig(out / "error_by_time.png", dpi=160)
    plt.close(fig)
