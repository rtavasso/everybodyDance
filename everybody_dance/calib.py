"""Calibration for the song-builder: derive a per-person Profile plus the two
extra knobs the builder needs -- a locked tempo (folded into a musical range) and
a per-person hit threshold for onset detection.

Batch (`calibrate_build`) is used offline; the streaming `BuildCalibrator` is used
live, accumulating while the dancer warms up and finalising on demand.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .features import FeatureExtractor
from .laban import LabanEstimator
from .oscillator import EntrainedClock
from .personalization import Calibrator, Profile
from .pose import JOINT_INDEX

HIT_JOINTS = ("l_wrist", "r_wrist", "l_ankle", "r_ankle")


def fold_tempo(tempo_hz: float, lo: float = 1.0, hi: float = 2.3) -> float:
    """Octave-fold a detected tempo into a comfortable musical band."""
    if tempo_hz <= 0:
        return 2.0
    while tempo_hz < lo:
        tempo_hz *= 2
    while tempo_hz > hi:
        tempo_hz /= 2
    return tempo_hz


class BuildCalibrator:
    """Streaming calibration; feed frames, then finalise()."""

    def __init__(self, fps: float = 30.0):
        self._cal = Calibrator()
        self._fe = FeatureExtractor(fps)
        self._lab = LabanEstimator()
        self._clk = EntrainedClock()
        self._idx = [JOINT_INDEX[j] for j in HIT_JOINTS]
        self._hits = []

    def observe(self, frame) -> None:
        f = self._fe.update(frame)
        self._cal.observe(f, self._lab.update(f, f.dt),
                          self._clk.update(float(f.bounce), f.dt).tempo_hz)
        self._hits.append(float(f.accel_mag[self._idx].max()))

    @property
    def n(self) -> int:
        return len(self._hits)

    def finalize(self) -> Tuple[Profile, float, float]:
        prof = self._cal.finalize(signature=self._fe.signature.summary())
        tempo = fold_tempo(prof.char_tempo_hz)
        hit_thr = float(np.percentile(self._hits, 72)) if self._hits else 8.0
        return prof, tempo, hit_thr


def calibrate_build(src) -> Tuple[Profile, float, float]:
    cal = BuildCalibrator(src.fps)
    for fr in src.frames():
        cal.observe(fr)
    return cal.finalize()
