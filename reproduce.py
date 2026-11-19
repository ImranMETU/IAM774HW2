"""Rebuild the benchmark and PINNs in an isolated source tree."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent


def run(directory, script, *args):
    subprocess.run([sys.executable, script, *args], cwd=directory, check=True)


def compare(directory):
    results = directory / "results"
    archived = ROOT / "results"
    comparisons = {}
    for file in sorted(results.rglob("*.npz")):
        relative = file.relative_to(results)
        with np.load(file) as actual, np.load(archived / relative) as expected:
            assert set(actual.files) == set(expected.files)
            comparisons[relative.as_posix()] = {}
            for name in actual.files:
                np.testing.assert_allclose(actual[name], expected[name], rtol=0, atol=1e-10,
                                           err_msg=f"{relative} {name}")
                comparisons[relative.as_posix()][name] = float(np.max(np.abs(actual[name] - expected[name])))
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".reproduction" / "replay")
    parser.add_argument("--full", action="store_true", help="Also rebuild all robustness runs")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for source in sorted(ROOT.glob("*.py")) + sorted(ROOT.glob("*config.json")) + [ROOT / "case.json"]:
        shutil.copy2(source, output / source.name)
    run(output, "benchmark.py")
    run(output, "train_forward.py")
    run(output, "train_inverse.py")
    if args.full:
        run(output, "train_forward.py", "--seed", "67508")
        run(output, "train_forward.py", "--seed", "67509")
        run(output, "train_forward.py", "--output", "results/forward/diagnostics-67507")
        run(output, "forward_robustness.py")
        run(output, "inverse_robustness.py")
        run(output, "summarize_inverse_robustness.py")
        for failed in (output / "results").rglob("failure.json"):
            raise RuntimeError(f"An experiment failed in the replay: {failed.relative_to(output)}")
    comparisons = compare(output)
    report = {"full": args.full, "max_absolute_array_differences": comparisons}
    (output / "replay_comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
