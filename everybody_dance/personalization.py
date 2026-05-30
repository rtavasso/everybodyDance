"""Personalisation -- required, not a feature.

Fixed mappings break across bodies. A short calibration captures each person's
range, characteristic tempo, energy distribution and Laban signature. Then:

  1. **normalise** every readout driver and every Effort factor onto *their*
     range, so their full movement maps to the full musical range -- this alone
     makes it feel alive for everyone, and (crucially on real data) stops Effort
     from saturating;
  2. **bias** the substrate from their absolute signature (gentle/slow -> ambient
     legato & a slower musical band; sharp/energetic -> rhythmic staccato),
     including scale, register and density bounds.

Emergent uniqueness (deterministic map + unique body = unique output) rides on
top for free.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .features import Features
from .laban import Effort
from .readout import NormFeatures

DRIVER_KEYS = ["weight", "time", "space", "flow"]


@dataclass
class Profile:
    # robust percentile ranges (p5, p95) for readout drivers...
    energy: List[float] = field(default_factory=lambda: [0.0, 1.0])
    core_energy: List[float] = field(default_factory=lambda: [0.0, 1.0])
    limb_energy: List[float] = field(default_factory=lambda: [0.0, 1.0])
    openness: List[float] = field(default_factory=lambda: [0.0, 1.0])
    com_height: List[float] = field(default_factory=lambda: [-1.0, 1.0])
    # ...and for each Effort driver
    effort: Dict[str, List[float]] = field(
        default_factory=lambda: {k: [0.0, 1.0] for k in DRIVER_KEYS})
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
    def __init__(self):
        self._energy: List[float] = []
        self._core: List[float] = []
        self._limb: List[float] = []
        self._openness: List[float] = []
        self._com: List[float] = []
        self._drivers: Dict[str, List[float]] = {k: [] for k in DRIVER_KEYS}
        self._tempos: List[float] = []

    def observe(self, f: Features, drivers: dict,
                tempo_hz: Optional[float] = None):
        if not f.present:
            return
        self._energy.append(f.energy_env)
        self._core.append(f.core_energy_env)
        self._limb.append(f.limb_energy_env)
        self._openness.append(f.openness)
        self._com.append(f.com_height)
        for k in DRIVER_KEYS:
            self._drivers[k].append(drivers[k])
        if tempo_hz is not None:
            self._tempos.append(tempo_hz)

    @staticmethod
    def _rng(xs, lo=5, hi=95, default=(0.0, 1.0)):
        if len(xs) < 5:
            return list(default)
        a, b = float(np.percentile(xs, lo)), float(np.percentile(xs, hi))
        if b - a < 1e-4:
            b = a + 1e-3
        return [a, b]

    def finalize(self, signature: dict | None = None) -> Profile:
        tempo = 2.0
        if self._tempos:
            tempo = float(np.median(self._tempos[len(self._tempos) // 3:]))
        sig = dict(signature or {})
        # absolute character used for substrate biasing
        sig.setdefault("energy_mean", float(np.mean(self._energy or [0.0])))
        sig.setdefault("weight_mean", float(np.mean(self._drivers["weight"] or [0.0])))
        sig.setdefault("openness_mean", float(np.mean(self._openness or [0.0])))
        return Profile(
            energy=self._rng(self._energy),
            core_energy=self._rng(self._core),
            limb_energy=self._rng(self._limb),
            openness=self._rng(self._openness),
            com_height=self._rng(self._com, default=(-1.0, 1.0)),
            effort={k: self._rng(self._drivers[k]) for k in DRIVER_KEYS},
            char_tempo_hz=tempo,
            signature=sig,
        )


class Normalizer:
    def __init__(self, profile: Profile):
        self.p = profile

    @staticmethod
    def _n(x: float, rng: List[float]) -> float:
        a, b = rng
        return float(np.clip((x - a) / (b - a + 1e-9), 0.0, 1.0))

    def effort(self, drivers: dict) -> Effort:
        return Effort(**{k: self._n(drivers[k], self.p.effort[k])
                         for k in DRIVER_KEYS})

    def features(self, f: Features, eff: Effort) -> NormFeatures:
        return NormFeatures(
            energy=self._n(f.energy_env, self.p.energy),
            core_energy=self._n(f.core_energy_env, self.p.core_energy),
            limb_energy=self._n(f.limb_energy_env, self.p.limb_energy),
            openness=self._n(f.openness, self.p.openness),
            com_height=self._n(f.com_height, self.p.com_height),
            weight=eff.weight,            # already per-person normalised
        )


def identity_profile() -> Profile:
    return Profile(energy=[0.0, 0.5], openness=[0.5, 2.0],
                   com_height=[-0.2, 0.2])
