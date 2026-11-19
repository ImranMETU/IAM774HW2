# Reproduction and release audit

The archived experiments use the illustrative case in [case.json](case.json). Settings for both PINNs are recorded in [pinn_config.json](pinn_config.json) and [inverse_config.json](inverse_config.json). Per-run directories retain the resolved settings, weights, losses, optimizer counts, wall-clock time, grid predictions, and figures. The inverse directories include the actual noisy observations and the least-squares fits.

## Clean environment

The verified environment is CPython 3.14.5, Windows x86-64, with PyTorch 2.15.0+cpu. NumPy, SciPy, Matplotlib, and pytest are direct dependencies. The CPU PINN requirements add PyTorch. The Windows constraints pin the exact direct and transitive versions from the clean install. Timing depends on CPU and machine load. Other platforms may require different CPU wheels and were outside this audit.

~~~powershell
py -3.14 -m venv .venv
$python = ".\.venv\Scripts\python.exe"
& $python -m pip install -r requirements-pinn.txt -c constraints-windows-cpu.txt
& $python -m pip check
& $python -m pytest -q
~~~

## Rebuild without overwriting the archive

Run these commands from the repository root. Use a new output path for each replay or figure check.

~~~powershell
& $python verify_release.py --output .reproduction/figure-check
& $python reproduce.py --output .reproduction/seed-67507
& $python reproduce.py --full --output .reproduction/all-runs
~~~

The first command reloads the checkpoints, recomputes archived metrics, checks loss normalization and optimizer counts, and regenerates the figures from the saved data. It reports pixel differences separately from numerical errors because rendering can depend on the font and graphics stack. The second builds the Crank–Nicolson benchmark and both seed-67507 PINNs in an isolated copy of the source. The third extends the replay to both robustness studies. The replay checks reconstructed numerical arrays with absolute tolerance 1e-10 and leaves observed wall-clock times in their run directories.

To run the individual entry points against the archived reference grid, use

~~~powershell
& $python benchmark.py --output .reproduction/baseline
& $python train_forward.py --seed 67507 --output .reproduction/forward-67507
& $python train_inverse.py --output .reproduction/inverse-67507
& $python inverse_robustness.py --output .reproduction/inverse-seed-study
& $python summarize_inverse_robustness.py --output .reproduction/inverse-seed-study
~~~

The forward PINN selects the network and training point streams with the run seed. The inverse robustness runner fixes the extra collocation stream, reuses the original archived observations for the initialization study, and draws new observations with NumPy for the observation study. The original inverse experiment is the shared baseline.

## Audit findings

The clean Windows environment passed all 20 tests and dependency checks. Reloading all nine checkpoints reproduced the archived grid predictions exactly. All 35 plots regenerated from the saved data with zero pixel differences in that environment. The isolated replay rebuilt the four-grid baseline and both seed-67507 PINNs from scratch, reproducing the archived reference, prediction and observation arrays exactly.

The replay changed wall-clock timings. On this replay the finest Crank–Nicolson solve took 48.42 ms median, forward training took 118.69 s, and inverse training took 103.41 s. The archived values were 28.30 ms, 92.86 s, and 83.18 s. The archive retains the original performance measurements. The complete robustness suite was checked from saved configurations, histories and checkpoints. Fresh training in this audit covered the two seed-67507 PINNs.

| Area | Finding |
|---|---|
| Paths and private data | Two archive metadata files were converted to repository-relative provenance. The writer now emits portable references. The results directory contains synthetic observations, public experiment seeds, and results from the independent project. |
| Dependencies | All direct libraries have an execution or test role. The clean install uses the CPU wheel index for PyTorch, and the release excludes virtual environments and installed packages. |
| RNG and precision | Training uses explicit CPU Torch generators and float64. Observation generation uses a separate NumPy Generator. Float64 checkpoint reloading and seed separation are tested. Replay is limited to the verified environment. |
| Loss and metrics | Each loss term averages the squared quantity over its own point set before applying its weight. The PINN field metric compares Euclidean norms on identical coordinates, and the inverse observation RMSE uses the exact readings seen during fitting. Signed parameter error is estimate divided by truth minus one. |
| Data leakage | The PINNs receive prescribed initial and boundary data, and the inverse PINN receives noisy measurements. Clean analytical field values are used for synthetic generation and evaluation. Forward held-out residual points were checked for overlap with every training batch. The independent field grid includes the prescribed boundary and initial slices. |
| Errors and recovery | A nonfinite training loss now retains the preceding available loss history. Forward training writes a failure record, as the inverse path already does. Run directories protect existing artifacts. Training cannot resume from optimizer state after abrupt interruption. |
| Provenance | Original settings, architectures and numerical equations were retained. Forward stopping diagnostics now identify whether the reason came from an original instrumented run or a separate replay. The archive contains the original seed-67507 replay used to establish its stopping reason. |
| Performance | Archived runtime figures are solve or training costs. PINN inference runtime was unmeasured. Numerical convergence does not imply bitwise reproducibility across platforms. |

## Release scope and licensing

The repository is scoped to the independent heat-transfer project. Instructor documents and parameter generators, unrelated coursework, blind datasets, student identifiers, dependency binaries, and local caches are excluded. The experiments use illustrative coefficients and independent synthetic data. All archived runs and the diagnostics replay are retained.

The MIT license covers the project files. The dependencies are installed separately and retain their upstream terms. NumPy and SciPy use BSD-style project licenses, PyTorch uses a BSD-style project license with its own bundled notices, Matplotlib uses its PSF-based license, and pytest uses MIT. The repository distributes no dependency wheels or source trees. [Project license](LICENSE) · [NumPy](https://numpy.org/doc/stable/license.html) · [SciPy](https://github.com/scipy/scipy/blob/main/LICENSE.txt) · [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE) · [Matplotlib](https://matplotlib.org/stable/project/license.html) · [pytest](https://github.com/pytest-dev/pytest/blob/main/LICENSE)
