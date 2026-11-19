"""Two-mode heat conduction on the unit interval with zero end temperatures."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cholesky_banded, cho_solve_banded


@dataclass(frozen=True)
class Rod:
    kappa: float
    loss_rate: float
    amplitude: float

    def __post_init__(self):
        if not np.all(np.isfinite([self.kappa, self.loss_rate, self.amplitude])):
            raise ValueError("Coefficients must be finite.")
        if self.kappa <= 0 or self.loss_rate < 0:
            raise ValueError("Expected positive diffusivity and nonnegative heat loss.")

    def decay_rate(self, mode: int) -> float:
        return self.kappa * (mode * np.pi) ** 2 + self.loss_rate

    def initial(self, x):
        x = np.asarray(x)
        return np.sin(np.pi * x) + self.amplitude * np.sin(4 * np.pi * x)

    def exact(self, x, t):
        """Broadcast x and t. For a time-by-space field pass x and t[:, None]."""
        x, t = np.asarray(x), np.asarray(t)
        first = np.sin(np.pi * x) * np.exp(-self.decay_rate(1) * t)
        fourth = np.sin(4 * np.pi * x) * np.exp(-self.decay_rate(4) * t)
        return first + self.amplitude * fourth


def crank_nicolson(rod: Rod, nx: int, nt: int):
    """Return x, t, and u[time, space]. nx and nt count intervals on [0, 1]."""
    if nx < 2 or nt < 1:
        raise ValueError("Need at least two spatial intervals and one time step.")
    x = np.linspace(0, 1, nx + 1)
    t = np.linspace(0, 1, nt + 1)
    u = np.zeros((nt + 1, nx + 1))
    u[0, 1:-1] = rod.initial(x[1:-1])

    # Half-step diffusion and reaction weights in I +/- dt*A/2.
    d = rod.kappa * nx**2 / (2 * nt)
    q = rod.loss_rate / (2 * nt)
    lhs = np.zeros((2, nx - 1))
    lhs[0] = 1 + 2 * d + q
    lhs[1, :-1] = -d
    factor = cholesky_banded(lhs, lower=True, check_finite=False)
    for step in range(nt):
        prev = u[step]
        rhs = (1 - 2 * d - q) * prev[1:-1] + d * (prev[:-2] + prev[2:])
        u[step + 1, 1:-1] = cho_solve_banded(
            (factor, True), rhs, check_finite=False
        )
    return x, t, u


def relative_l2(prediction, reference) -> float:
    """Euclidean norm ratio over all sample points, with equal weights."""
    prediction, reference = np.asarray(prediction), np.asarray(reference)
    if prediction.shape != reference.shape:
        raise ValueError("Prediction and reference must have identical shapes.")
    energy = np.sum(reference**2)
    if energy == 0:
        raise ValueError("Relative error requires a nonzero reference.")
    return float(np.sqrt(np.sum((prediction - reference) ** 2) / energy))
