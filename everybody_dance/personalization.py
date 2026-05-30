"""Personalisation -- required, not a feature.

Fixed mappings break across bodies: tall vs short, expansive vs subtle movers
hit completely different feature ranges, so one mapping feels dead for half of
people. A short calibration ("move freely ~20s") captures each person's range,
characteristic frequency, energy distribution and Laban signature. Then:

  1. normalise so their full range maps to the full musical range -- this alone
     makes it feel alive for everyone;
  2. bias the substrate from their signature (flowing/slow -> ambient legato &
     slower tempo band; sharp/percussive -> rhythmic staccato).

Emergent uniqueness (deterministic map + unique body = unique output) rides on
top for free. The learned per-person embedding is the heavy path -- skipped here.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .features import Features
from .laban import Effort
from .readout import NormFeatures


@dataclass
class Profile:
    # robust percentile ranges per driver (p5, p95)
    energy: List[float] = field(default_factory=lambda: [0.0, 1.0])
    openness: List[float] = field(default_factory=lambda: [0.0, 1.0])
    com_height: List[float] = field(default_factory=lambda: [-1.0, 1.0])
    weight: List[float] = field(default_factory=lambda: [0.0, 1.0])
    char_tempo_hz: float = 2.0
    signature: Dict[str, float] = field(default_factory=dict)

    def to_json(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)

    @staticmethod
    def from_json(path: str) -> "Profile":
        with open(path) as fh:
            return Profile(**json.load(fh))


class Calibrator:
    """Collects feature/effort samples during the free-movement window."""

    def __init__(self):
        self._energy: List[float] = []
        self._openness: List[float] = []
        self._com: List[float] = []
        self._weight: List[float] = []
        self._tempos: List[float] = []

    def observe(self, f: Features, eff: Effort, tempo_hz: Optional[float] = None):
        if not f.present:
            return
        self._energy.append(f.energy_env)
        self._openness.append(f.openness)
        self._com.append(f.com_height)
        self._weight.append(eff.weight)
        if tempo_hz is not None:
            self._tempos.append(tempo_hz)

    def finalize(self, signature: dict | None = None) -> Profile:
        def rng(xs, lo=5, hi=95, default=(0.0, 1.0)):
            if len(xs) < 5:
                return list(default)
            a = float(np.percentile(xs, lo))
            b = float(np.percentile(xs, hi))
            if b - a < 1e-4:
                b = a + 1e-3
            return [a, b]

        tempo = 2.0
        if self._tempos:
            tempo = float(np.median(self._tempos[len(self._tempos) // 3:]))
        return Profile(
            energy=rng(self._energy),
            openness=rng(self._openness),
            com_height=rng(self._com, default=(-1.0, 1.0)),
            weight=rng(self._weight),
            char_tempo_hz=tempo,
            signature=signature or {},
        )


class Normalizer:
    """Maps raw drivers into [0,1] using the calibrated ranges."""

    def __init__(self, profile: Profile):
        self.p = profile

    @staticmethod
    def _n(x: float, rng: List[float]) -> float:
        a, b = rng
        return float(np.clip((x - a) / (b - a + 1e-9), 0.0, 1.0))

    def __call__(self, f: Features, eff: Effort) -> NormFeatures:
        return NormFeatures(
            energy=self._n(f.energy_env, self.p.energy),
            openness=self._n(f.openness, self.p.openness),
            com_height=self._n(f.com_height, self.p.com_height),
            weight=self._n(eff.weight, self.p.weight),
        )


def identity_profile() -> Profile:
    """A neutral profile for running without calibration."""
    return Profile(energy=[0.0, 0.5], openness=[0.5, 2.0],
                   com_height=[-0.2, 0.2], weight=[0.0, 1.0])
