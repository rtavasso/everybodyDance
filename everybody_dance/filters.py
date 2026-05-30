"""Low-latency smoothing primitives.

The One-Euro filter (Casiet, Roussel, Vogel 2012) is the right tool for
interactive motion: it adapts its cutoff to speed, killing jitter when still
without adding lag when moving fast. We allocate smoothing *by timescale* --
the fast lane runs nearly raw, the slow lane buffers harder.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Scalar (or vector, element-wise) one-euro filter.

    Parameters
    ----------
    min_cutoff : baseline cutoff (Hz). Lower = smoother but laggier when slow.
    beta       : speed coefficient. Higher = less lag when moving fast.
    d_cutoff   : cutoff for the derivative estimate.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0,
                 d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: Optional[np.ndarray] = None
        self._dx_prev: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = None

    def __call__(self, x, dt: float):
        x = np.asarray(x, dtype=float)
        if self._x_prev is None:
            self._x_prev = x
            self._dx_prev = np.zeros_like(x)
            return x
        dt = max(dt, 1e-4)
        # Derivative, smoothed at d_cutoff.
        dx = (x - self._x_prev) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev
        # Speed-adaptive cutoff.
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = _alpha_vec(cutoff, dt)
        x_hat = a * x + (1 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


def _alpha_vec(cutoff, dt: float):
    tau = 1.0 / (2.0 * np.pi * np.asarray(cutoff, dtype=float))
    return 1.0 / (1.0 + tau / dt)


class EMA:
    """Plain exponential moving average with a time constant in seconds."""

    def __init__(self, tau: float):
        self.tau = float(tau)
        self._y = None

    def __call__(self, x, dt: float):
        x = np.asarray(x, dtype=float)
        if self._y is None:
            self._y = x
            return x
        a = 1.0 - math.exp(-dt / max(self.tau, 1e-4))
        self._y = a * x + (1 - a) * self._y
        return self._y

    @property
    def value(self):
        return self._y
