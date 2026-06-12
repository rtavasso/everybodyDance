"""Groove -- the musical backbone that makes the Studio *tight*.

The coupling lesson from real video: mapping continuous movement straight into
per-step pattern recomputation destroys music. A beat only exists if it
REPEATS; a melody only exists if it returns. So the body stops micro-managing
note placement and instead *commits musical structure at musical boundaries*:

  * :class:`TempoLatch` -- tempo is estimated from the dance but LATCHED to an
    integer BPM; it re-locks only after a sustained, stable deviation, and only
    at a bar line. Between re-locks the grid is metronomic and notes are
    scheduled at exact grid times.
  * :data:`STYLES` -- curated drum grooves (kick/snare/hat/ghost over a 16-step
    bar, four density levels each, real backbeats). The dancer's identity picks
    the style; the bar's *level* is committed from movement at the bar line;
    velocity keeps following the body continuously. Bass locks to the kick on
    chord roots; keys comp on style slots.
  * :class:`MotifWriter` -- the lead is a short motif the body authors (rhythm
    template + pitches sampled from the dancer's height contour, chord tones on
    strong slots, stepwise elsewhere), which persists across bars, re-roots
    with the harmony, and mutates one slot at a time. Repetition + variation.
  * :class:`RhythmCoherence` -- the skill loop: how consistently the dancer's
    bounce peaks land on the latched grid (circular statistics). Low coherence
    keeps the kit sparse and plain; locking to the beat earns the backbeat,
    ghosts, the full motif, and feeds the song arc.

Everything is deterministic: same dance -> same tempo locks, same plans, same
motif, same score.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

# -- tempo -------------------------------------------------------------------


class TempoLatch:
    """Integer-BPM tempo lock with deliberate, sustained re-locking.

    `observe` feeds the raw entrained tempo estimate every frame; `proposal`
    returns a new integer BPM only when the folded estimate has been BOTH far
    from the current lock (> `dev`) AND internally stable for `hold_s`. The
    caller re-locks at a bar line via `relock`, so tempo changes land as a
    musical event, never a slide."""

    def __init__(self, initial_hz: float = 2.0, band: Tuple[float, float] = (80.0, 160.0),
                 dev: float = 0.08, hold_s: float = 3.0):
        self.band = band
        self.dev = dev
        self.hold_s = hold_s
        self.bpm = float(int(round(self.fold((initial_hz or 2.0) * 60.0))))
        self.relocks = 0
        self._est: deque = deque()

    def fold(self, bpm: float) -> float:
        lo, hi = self.band
        bpm = float(max(bpm, 1e-3))
        for _ in range(8):
            if bpm < lo:
                bpm *= 2.0
            elif bpm > hi:
                bpm /= 2.0
            else:
                break
        return float(np.clip(bpm, lo, hi))

    @property
    def beat_s(self) -> float:
        return 60.0 / self.bpm

    def observe(self, tempo_hz: float, t: float) -> None:
        self._est.append((t, self.fold(tempo_hz * 60.0)))
        while self._est and t - self._est[0][0] > self.hold_s:
            self._est.popleft()

    def proposal(self) -> Optional[float]:
        if len(self._est) < 10:
            return None
        es = np.array([e for _, e in self._est])
        med = float(np.median(es))
        if abs(med - self.bpm) / self.bpm <= self.dev:
            return None
        # mid-slide estimates are noisy; only a settled estimate may re-lock.
        if (np.percentile(es, 75) - np.percentile(es, 25)) / med > 0.06:
            return None
        return float(int(round(med)))

    def relock(self, bpm: float) -> None:
        self.bpm = float(bpm)
        self.relocks += 1
        self._est.clear()

    def trim(self, drift_beats_per_beat: float) -> None:
        """A fine tempo correction from the phase servo's integral term: the
        grid kept shifting `drift` beats later per beat, so the true period is
        (1 + drift) longer. Sub-integer, silent, not counted as a re-lock."""
        lo, hi = self.band
        self.bpm = float(np.clip(self.bpm / (1.0 + drift_beats_per_beat),
                                 lo, hi))


# -- drum grooves --------------------------------------------------------------

# 16-step bars (4 steps per beat). Four density LEVELS per style: 0 = heartbeat,
# 1 = core groove, 2 = full groove with backbeat, 3 = + ghosts/16ths. Curated --
# every level of every style is a pattern a drummer would actually play.
STYLES: Dict[str, List[Dict[str, List[int]]]] = {
    "four_floor": [
        {"kick": [0, 8], "snare": [], "hat": [4, 12], "ghost": []},
        {"kick": [0, 4, 8, 12], "snare": [], "hat": [2, 6, 10, 14], "ghost": []},
        {"kick": [0, 4, 8, 12], "snare": [4, 12], "hat": [2, 6, 10, 14],
         "ghost": []},
        {"kick": [0, 4, 8, 12], "snare": [4, 12], "hat": [2, 6, 10, 14],
         "ghost": [1, 5, 9, 13]},
    ],
    "backbeat": [
        {"kick": [0, 8], "snare": [], "hat": [0, 8], "ghost": []},
        {"kick": [0, 8], "snare": [4, 12], "hat": [0, 4, 8, 12], "ghost": []},
        {"kick": [0, 8, 10], "snare": [4, 12], "hat": [0, 2, 4, 6, 8, 10, 12, 14],
         "ghost": []},
        {"kick": [0, 8, 10], "snare": [4, 12], "hat": [0, 2, 4, 6, 8, 10, 12, 14],
         "ghost": [3, 7, 11, 15]},
    ],
    "breaks": [
        {"kick": [0, 10], "snare": [], "hat": [4, 12], "ghost": []},
        {"kick": [0, 10], "snare": [4, 12], "hat": [0, 4, 8, 12], "ghost": []},
        {"kick": [0, 6, 10], "snare": [4, 12], "hat": [0, 2, 4, 6, 8, 10, 12, 14],
         "ghost": []},
        {"kick": [0, 6, 10], "snare": [4, 12], "hat": [0, 2, 4, 6, 8, 10, 12, 14],
         "ghost": [7, 15]},
    ],
    "half_time": [
        {"kick": [0], "snare": [], "hat": [8], "ghost": []},
        {"kick": [0, 10], "snare": [8], "hat": [0, 4, 8, 12], "ghost": []},
        {"kick": [0, 10], "snare": [8], "hat": [0, 2, 4, 6, 8, 10, 12, 14],
         "ghost": []},
        {"kick": [0, 7, 10], "snare": [8], "hat": [0, 2, 4, 6, 8, 10, 12, 14],
         "ghost": [5, 13]},
    ],
}

# Style -> swing: how late the odd 16ths sit, as a fraction of a step.
SWING = {"four_floor": 0.0, "backbeat": 0.10, "breaks": 0.14, "half_time": 0.18}

# Base velocities per drum lane (live energy/weight modulates on top).
LANE_VEL = {"kick": 98, "snare": 88, "hat": 56, "ghost": 32}

# The fill overlay: a snare run up the last quarter of the bar + driving hats.
FILL_SNARE = [12, 13, 14, 15]
FILL_HAT = [0, 2, 4, 6, 8, 10]

# Keys comping slots per style (level >= 2 plays stabs; below it, a held pad).
COMP_SLOTS = {"four_floor": [2, 6, 10, 14], "backbeat": [4, 12],
              "breaks": [2, 10], "half_time": [0, 8]}


def drum_plan(style: str, level: int, fill: bool = False) -> List[List[Tuple[str, int]]]:
    """The committed drum bar: 16 slots of (piece-lane, base velocity)."""
    pat = STYLES[style][int(np.clip(level, 0, 3))]
    plan: List[List[Tuple[str, int]]] = [[] for _ in range(16)]
    for lane in ("kick", "snare", "hat", "ghost"):
        for s in pat[lane]:
            piece = "hat" if lane == "ghost" else lane
            plan[s].append((piece, LANE_VEL[lane]))
    if fill:
        for k, s in enumerate(FILL_SNARE):
            plan[s].append(("snare", 64 + 10 * k))
        for s in FILL_HAT:
            if not any(p == "hat" for p, _ in plan[s]):
                plan[s].append(("hat", LANE_VEL["hat"]))
    return plan


def bass_plan(style: str, level: int) -> List[List[Tuple[int, int, float]]]:
    """Bass locks to the kick: (scale-degree offset from chord root, base vel,
    dur in beats) per slot. Level 2 adds the fifth, level 3 an octave pickup."""
    pat = STYLES[style][int(np.clip(level, 0, 3))]
    plan: List[List[Tuple[int, int, float]]] = [[] for _ in range(16)]
    for s in pat["kick"]:
        plan[s].append((0, 92, 0.45))                  # the root, on the kick
    if level >= 2:
        off = 6 if not plan[6] else 5
        plan[off].append((4, 78, 0.30))                # the fifth, off the beat
    if level >= 3:
        plan[14].append((7, 74, 0.22))                 # octave pickup into 1
    return plan


def keys_plan(style: str, level: int) -> List[Tuple[int, str]]:
    """Keys: a held pad on the bar at low levels, style-slot stabs above.
    Returns (slot, kind) with kind in {'pad','stab'}."""
    if level <= 1:
        return [(0, "pad")]
    return [(s, "stab") for s in COMP_SLOTS[style]]


def texture_plan(level: int, bar_idx: int) -> List[int]:
    """The polyrhythmic texture: 3- or 5-pulse euclidean against the 4/4,
    rotated every four bars so it slowly orbits the kit."""
    from .substrate import euclidean
    if level <= 0:
        return []
    pulses = 3 if level < 3 else 5
    base = euclidean(pulses, 16)
    rot = ((bar_idx // 4) % 4) * 2
    pat = base[rot:] + base[:rot]
    return [s for s in range(16) if pat[s]]


# -- the motif engine ------------------------------------------------------------

# Rhythm templates per level: which 16th slots the motif sings on. Strong slots
# (on the beat) carry chord tones; the rest move stepwise. Curated, singable.
MOTIF_TEMPLATES: Dict[int, List[List[int]]] = {
    0: [[0, 8]],
    1: [[0, 6, 8], [0, 4, 10], [0, 8, 12]],
    2: [[0, 4, 6, 10], [0, 3, 8, 11], [0, 6, 8, 14], [0, 4, 8, 10, 12]],
    3: [[0, 3, 6, 8, 11, 14], [0, 2, 4, 8, 10, 12], [0, 4, 7, 8, 12, 14]],
}
CHORD_OFFSETS = (0, 2, 4, 7)        # chord tones (+ octave root), scale degrees


class MotifWriter:
    """The body authors a motif; the system keeps it, develops it, re-roots it.

    `maybe_compose` runs at bar lines: a NEW motif only when the level bucket
    changed or every 8 bars (identity), with a single-slot mutation every 4
    bars (variation). Pitches sample the dancer's recent height contour --
    raise your body and the next phrase sits higher -- snapped to chord tones
    on strong slots and held to stepwise motion elsewhere, so it always sings.
    Playback realizes degrees against the CURRENT chord root, so the same
    motif follows the harmonic field (and THE LIFT) for free."""

    def __init__(self, seed: int = 0):
        self.seed = int(seed)
        self.slots: List[Tuple[int, int, bool]] = []   # (slot, degree off, strong)
        self._level = -1

    def maybe_compose(self, level: int, contour: List[float], bar_idx: int) -> None:
        level = int(np.clip(level, 0, 3))
        if self.slots and level == self._level and bar_idx % 8 != 0:
            if bar_idx % 4 == 0:
                self._mutate(contour, bar_idx)
            return
        self._level = level
        tpl = MOTIF_TEMPLATES[level]
        template = tpl[(self.seed + bar_idx // 8) % len(tpl)]
        self.slots = []
        prev = 4
        for s in template:
            off = self._offset_at(s, contour, prev, strong=(s % 4 == 0))
            self.slots.append((s, off, s % 4 == 0))
            prev = off

    def _offset_at(self, slot: int, contour: List[float], prev: int,
                   strong: bool) -> int:
        h = contour[min(int(slot / 16.0 * len(contour)), len(contour) - 1)] \
            if contour else 0.5
        off = int(round(float(np.clip(h, 0, 1)) * 7))
        # stepwise constraint: never leap more than 3 scale steps at once.
        off = int(np.clip(off, prev - 3, prev + 3))
        if strong:
            off = min(CHORD_OFFSETS, key=lambda c: abs(c - off))
        return int(off)

    def _mutate(self, contour: List[float], bar_idx: int) -> None:
        k = (self.seed + bar_idx // 4) % len(self.slots)
        s, _, strong = self.slots[k]
        prev = self.slots[k - 1][1] if k else 4
        self.slots[k] = (s, self._offset_at(s, contour, prev, strong), strong)

    def plan(self, chord_root_degree: int) -> List[Tuple[int, int, bool]]:
        """(slot, absolute scale degree, strong) for the current chord."""
        return [(s, chord_root_degree + off, strong)
                for s, off, strong in self.slots]


# -- the body as an audio signal -----------------------------------------------


class KineticFlux:
    """Audio-style onset detection on the BODY: a rectified multi-joint
    acceleration envelope with an adaptive threshold (the 'visual beats' idea
    -- Davis & Agrawala 2018 -- applied to skeleton kinematics). A punch, a
    stomp, an arm hit, a plant: each is an accel spike that pops out of the
    dancer's own running statistics. Returns an onset strength (0..1) on the
    frame an accent lands, else None. Deterministic, streaming, O(joints)."""

    REFRACTORY = 0.18      # s -- faster than any human double-hit we care about
    Z_ON = 2.2             # threshold in adaptive std units

    # joint weights: extremities carry the accents (index = pose.JOINTS order:
    # nose, l/r shoulder, l/r elbow, l/r wrist, l/r hip, l/r knee, l/r ankle)
    W = np.array([0.2, 0.3, 0.3, 0.6, 0.6, 1.0, 1.0, 0.4, 0.4, 0.6, 0.6,
                  1.0, 1.0])

    def __init__(self):
        self._mean = 0.0
        self._var = 1.0
        self._prev = 0.0
        self._last_t = -1e9

    def update(self, accel_mag: np.ndarray, t: float, dt: float
               ) -> Optional[float]:
        flux = float((self.W[: len(accel_mag)] * accel_mag).mean())
        # adaptive stats (~2 s time constant) of the flux itself.
        k = float(np.clip(dt / 2.0, 0.0, 1.0))
        self._mean += k * (flux - self._mean)
        self._var += k * ((flux - self._mean) ** 2 - self._var)
        std = max(np.sqrt(self._var), 1e-4)
        z = (flux - self._mean) / std
        rising = flux > self._prev
        self._prev = flux
        if z > self.Z_ON and rising and t - self._last_t >= self.REFRACTORY:
            self._last_t = t
            return float(np.clip((z - self.Z_ON) / 4.0 + 0.25, 0.0, 1.0))
        return None


class PhaseServo:
    """The PLL that puts the downbeat ON the dancer's accents.

    Tempo is the latch's job; this owns PHASE. Each accent's signed error to
    the nearest HALF-beat (mod-8th: a dancer accenting on 8ths is on the grid)
    enters a weighted circular buffer; when the errors agree (resultant R >=
    0.5) the grid is nudged by up to 4% of a beat per beat -- inaudible per
    step, converging in a couple of bars -- with one full hard snap allowed
    early in a session (the band starts ON your hit). Incoherent accents (R
    low) are never chased: the grid holds steady."""

    PERIOD = 1.0           # beats: true downbeat alignment. A dancer accenting
                           # at 8th rate splits the circular mean (R drops) and
                           # is simply not chased -- they're already grid-dense.
    MAX_STEP = 0.04        # beats of correction per beat
    R_MIN = 0.5

    def __init__(self, window: int = 8):
        self._errs: deque = deque(maxlen=window)   # (err in beats, weight)
        self._corrs: deque = deque(maxlen=8)       # applied, in beats
        self.snapped = False

    def observe(self, t: float, strength: float, beat_s: float,
                next_beat_t: float) -> None:
        d = (t - next_beat_t) / beat_s
        e = (((d / self.PERIOD + 0.5) % 1.0) - 0.5) * self.PERIOD
        self._errs.append((float(e), float(max(strength, 1e-3))))

    def _mean_err(self) -> Tuple[float, float]:
        """Weighted circular mean error (beats) + resultant R (agreement)."""
        if len(self._errs) < 3:
            return 0.0, 0.0
        ang = np.array([2 * np.pi * e / self.PERIOD for e, _ in self._errs])
        w = np.array([wt for _, wt in self._errs])
        v = (w * np.exp(1j * ang)).sum() / w.sum()
        return float(np.angle(v) / (2 * np.pi) * self.PERIOD), float(np.abs(v))

    def correction(self, beat_s: float, allow_snap: bool) -> float:
        """Seconds to add to the upcoming grid (call once per beat)."""
        m, r = self._mean_err()
        if r < self.R_MIN:
            return 0.0
        if allow_snap and not self.snapped and abs(m) > self.MAX_STEP:
            corr = m * beat_s                       # one hard snap, early
            self.snapped = True
        else:
            corr = float(np.clip(m, -self.MAX_STEP, self.MAX_STEP)) * beat_s
        # the stored errors were measured against the old grid: re-base them.
        self._errs = deque(((e - corr / beat_s, w) for e, w in self._errs),
                           maxlen=self._errs.maxlen)
        self._corrs.append(corr / beat_s)
        return corr

    def persistent_drift(self) -> Optional[float]:
        """The PLL's integral term: if the servo keeps correcting the SAME way
        every beat, the latched tempo itself is slightly off -- return the
        mean drift (beats per beat) so the latch can trim, else None."""
        if len(self._corrs) < self._corrs.maxlen:
            return None
        c = np.array(self._corrs)
        if np.all(c > 0) or np.all(c < 0):
            m = float(c.mean())
            if abs(m) > 0.008:
                self._corrs.clear()
                return m
        return None


# -- the skill loop ----------------------------------------------------------------


class RhythmCoherence:
    """The pulse listener: detects the dancer's bounce peaks, and from them
    derives BOTH the tempo estimate (median inter-peak interval -- far more
    robust than the oscillator on real, half-rectified bounce signals) and the
    skill score (circular statistics of peak phases against the latched grid:
    a dancer pulsing with the tempo scores ~1, arrhythmic flailing ~0)."""

    NEUTRAL = 0.45
    PROMINENCE = 0.015           # torso units a peak must rise above its valley

    def __init__(self, window: int = 8):
        self.score = self.NEUTRAL
        self._phases: deque = deque(maxlen=window)
        self._peak_times: deque = deque(maxlen=window)
        self._filt = None
        self._prev = None
        self._prev2 = None
        self._rising = False
        self._last_peak_t = -1e9
        self._low = None

    def update(self, bounce: float, t: float, beat_s: float, dt: float,
               anchor_t: float = 0.0) -> float:
        # smooth first (~3 Hz one-pole): pose noise makes micro-peaks that the
        # raw signal's state machine would mistake for pulses. `anchor_t` is a
        # known grid-beat time, so phases stay honest when the servo shifts
        # the grid.
        raw = float(bounce)
        if self._filt is None:
            self._filt = raw
        k = dt / (dt + 1.0 / (2 * np.pi * 3.0))
        self._filt += k * (raw - self._filt)
        b = self._filt
        eps = 0.2 * self.PROMINENCE
        if self._low is None:
            self._low = b
        if self._prev is not None:
            if b > self._prev + eps:
                self._rising = True
            elif (self._rising and b < self._prev - eps
                    and t - self._last_peak_t > max(0.45 * beat_s, 0.25)
                    and self._prev - self._low > self.PROMINENCE):
                self._rising = False
                # sub-frame peak time: parabola through the three samples
                # around the maximum (frame quantisation would bias the tempo).
                tp = t - dt
                if self._prev2 is not None:
                    den = self._prev2 - 2.0 * self._prev + b
                    if abs(den) > 1e-12:
                        tp += float(np.clip(0.5 * (self._prev2 - b) / den,
                                            -0.5, 0.5)) * dt
                self._last_peak_t = t
                self._phases.append(((tp - anchor_t) / beat_s) % 1.0)
                self._peak_times.append(tp)
                self._low = b
            if not self._rising:
                self._low = min(self._low, b)
        self._prev2 = self._prev
        self._prev = b
        if len(self._phases) >= 4:
            v = np.exp(2j * np.pi * np.array(self._phases))
            target = float(np.abs(v.mean()))
        else:
            target = self.NEUTRAL
        self.score += float(np.clip(dt / 2.5, 0, 1)) * (target - self.score)
        return self.score

    @property
    def last_peak_t(self) -> Optional[float]:
        """Sub-frame time of the most recent bounce peak (the body's pulse)."""
        return self._peak_times[-1] if self._peak_times else None

    @property
    def tempo_hz(self) -> Optional[float]:
        """The dancer's pulse from inter-peak intervals, or None if too few."""
        if len(self._peak_times) < 4:
            return None
        med = float(np.median(np.diff(np.array(self._peak_times))))
        if not 0.2 < med < 2.5:
            return None
        return 1.0 / med

    def gate_level(self, level: int) -> int:
        """Skill gate: loose dancing keeps the kit plain (never broken)."""
        if self.score < 0.35:
            return min(level, 1)
        if self.score < 0.6:
            return min(level, 2)
        return level
