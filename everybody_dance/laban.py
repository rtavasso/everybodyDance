"""Laban Effort as the intermediate representation.

We deliberately do *not* map joints to notes. We map kinematics -> Laban Effort
(Weight, Time, Space, Flow) -> music. Effort is perceptually meaningful: a human
watching the dancer would route effort to music the same way, so the coupling
reads as natural "for free". This is where legibility comes from.

Each Effort factor is reported in [0, 1]:
  Weight  0=light      1=strong   (force / kinetic energy)
  Time    0=sustained  1=sudden   (acceleration burstiness)
  Space   0=indirect   1=direct   (path straightness)
  Flow    0=bound      1=free     (continuity of motion)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .features import Features
from .filters import EMA
from .pose import JOINT_INDEX, MASS_VEC


@dataclass
class Effort:
    weight: float
    time: float
    space: float
    flow: float

    def as_dict(self) -> dict:
        return {"weight": self.weight, "time": self.time,
                "space": self.space, "flow": self.flow}


def _sat(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


class LabanEstimator:
    """Turns a stream of :class:`Features` into a smoothed Effort vector.

    Scale constants are deliberate, rough priors; per-person calibration
    (see :mod:`personalization`) is what actually makes the ranges fit a body.
    """

    def __init__(self, window: int = 12):
        self._w = EMA(tau=0.25)
        self._t = EMA(tau=0.20)
        self._s = EMA(tau=0.45)   # Space is a phrase-ish quality -> slower
        self._f = EMA(tau=0.30)
        # COM path history for the Space (directness) computation.
        self._path = deque(maxlen=window)
        # velocity-continuity history for Flow.
        self._cont = deque(maxlen=window)

    def update(self, f: Features, dt: float) -> Effort:
        # --- Weight: strong vs light, from kinetic energy (force-like). ---
        weight = self._w(np.tanh(2.0 * f.kinetic_energy), dt)

        # --- Time: sudden vs sustained, from acceleration burstiness. ---
        # Sudden motion = high acceleration relative to speed (sharp changes).
        sp = float(f.speed.mean()) + 1e-3
        burst = float(f.accel_mag.mean()) / sp
        time = self._t(np.tanh(0.15 * burst), dt)

        # --- Space: direct vs indirect, COM path straightness. ---
        self._path.append(f.com.copy())
        space_raw = 0.5
        if len(self._path) >= 3:
            pts = np.array(self._path)
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
            net = np.linalg.norm(pts[-1] - pts[0])
            if seg > 1e-4:
                space_raw = net / seg     # 1 = straight/direct, ~0 = wandering
        space = self._s(space_raw, dt)

        # --- Flow: free vs bound, continuity of motion. ---
        # Free = velocity flows smoothly (low jerk relative to speed).
        # Bound = controlled, frequent checks (high jerk / start-stop).
        jerkiness = float(f.jerk_mag.mean()) / (sp + 1e-3)
        cont = 1.0 / (1.0 + 0.05 * jerkiness)
        # If nearly still, Flow is undefined; relax toward neutral-bound.
        if f.energy_env < 0.02:
            cont = 0.3
        flow = self._f(cont, dt)

        return Effort(_sat(weight), _sat(time), _sat(space), _sat(flow))
