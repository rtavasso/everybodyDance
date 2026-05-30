"""Laban Effort as the intermediate representation.

We map kinematics -> Laban Effort (Weight, Time, Space, Flow) -> music, because
Effort is perceptually meaningful: a human watching would route it to music the
same way, so the coupling reads as natural for free.

This estimator emits *raw* (smoothed but un-squashed) Effort drivers. The
per-person normaliser (see :mod:`personalization`) maps each driver onto that
body's own range, so Effort spans [0,1] for everyone -- alive, never saturated,
and comparable across bodies. (Earlier we squashed with hand-tuned constants;
those broke on real mocap, whose kinematic scales differ wildly from synthetic.)

Raw driver semantics (monotonic; the normaliser handles scale):
  weight_raw  kinetic energy            -> 0 light  .. 1 strong
  time_raw    accel / speed (burstiness)-> 0 sustained .. 1 sudden
  space_raw   path directness (net/arc) -> 0 indirect .. 1 direct
  flow_raw    -jerk/speed (smoothness)  -> 0 bound  .. 1 free
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .features import Features
from .filters import EMA


@dataclass
class Effort:
    """Normalised Effort, each factor in [0,1]."""
    weight: float
    time: float
    space: float
    flow: float

    def as_dict(self) -> dict:
        return {"weight": self.weight, "time": self.time,
                "space": self.space, "flow": self.flow}


class LabanEstimator:
    def __init__(self, window: int = 12):
        self._w = EMA(tau=0.25)
        self._t = EMA(tau=0.20)
        self._s = EMA(tau=0.45)   # Space reads as a phrase-ish quality -> slower
        self._f = EMA(tau=0.30)
        self._path = deque(maxlen=window)

    def update(self, f: Features, dt: float) -> dict:
        """Return raw Effort drivers (smoothed, un-normalised)."""
        sp = float(f.speed.mean()) + 1e-3

        weight_raw = float(self._w(f.kinetic_energy, dt))
        time_raw = float(self._t(float(f.accel_mag.mean()) / sp, dt))

        # Space: directness of the COM path over a short window.
        self._path.append(f.com.copy())
        directness = 0.5
        if len(self._path) >= 3:
            pts = np.array(self._path)
            arc = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
            net = np.linalg.norm(pts[-1] - pts[0])
            if arc > 1e-4:
                directness = net / arc
        space_raw = float(self._s(directness, dt))

        # Flow: smoothness (negative jerk-per-speed); more negative = more bound.
        jerkiness = float(f.jerk_mag.mean()) / sp
        flow_raw = float(self._f(-jerkiness, dt))

        return {"weight": weight_raw, "time": time_raw,
                "space": space_raw, "flow": flow_raw}
