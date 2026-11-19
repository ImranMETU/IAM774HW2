"""Run paired inverse experiments while preserving the original baseline."""

import argparse
import json
from pathlib import Path

from train_inverse import ROOT, run_experiment, write_json


SEEDS = (67507, 67508, 67509)
BASELINE = ROOT / "results" / "inverse" / "seed-67507"
DEFAULT_OUTPUT = BASELINE.parent / "robustness"


def experiment_plan(base, output, baseline=BASELINE):
    for experiment in ("initialization", "observation_noise"):
        for seed in SEEDS:
            config = {
                **base,
                "seed": seed if experiment == "initialization" else SEEDS[0],
                "observation_seed": SEEDS[0] if experiment == "initialization" else seed,
                "collocation_seed": SEEDS[0],
            }
            source = baseline / "observations.npz" if config["observation_seed"] == SEEDS[0] else None
            directory = baseline if seed == SEEDS[0] else output / experiment / f"seed-{seed}"
            yield experiment, seed, config, directory, source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    base = json.loads((ROOT / "inverse_config.json").read_text())
    for experiment, seed, config, directory, source in experiment_plan(base, args.output):
        if (directory / "metrics.json").is_file():
            print(f"{experiment} {seed} | using saved metrics", flush=True)
            continue
        if directory == BASELINE:
            raise FileNotFoundError("The original inverse baseline is required.")
        if directory.exists():
            print(f"{experiment} {seed} | preserving existing incomplete attempt", flush=True)
            continue
        print(f"{experiment} {seed} | training", flush=True)
        try:
            run_experiment(config, directory, source)
        except Exception as error:
            directory.mkdir(parents=True, exist_ok=True)
            write_json(directory / "failure.json", {"type": type(error).__name__, "message": str(error)})
            print(f"{experiment} {seed} | {type(error).__name__}: {error}", flush=True)


if __name__ == "__main__":
    main()
