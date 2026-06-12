"""Coherence substrate -- musicality by construction.

This is the safety net that makes "most movement sounds musical" true
regardless of input. The body drives *readout indices and modulation lanes*;
the substrate keeps everything in bounds:

  * a fixed scale / mode plus a slowly-moving harmonic field
  * per-track roles (bass / chord / lead / drums) with register & density bounds
  * euclidean rhythm patterns on a quantized grid -- sensible across the entire
    input range, so a feature can drive "pulses" raw and never make garbage
  * pitch snapped to the active scale, chord tones weighted

No looping is required: the entrained clock provides meter and phrase, so the
output has structure without literal periodicity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

# Scale degrees as semitone offsets from the tonic.
SCALES: Dict[str, List[int]] = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "minor_pentatonic": [0, 3, 5, 7, 10],
    "major_pentatonic": [0, 2, 4, 7, 9],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
}

# A gentle, mostly-diatonic harmonic field: scale-degree roots (0-based) that
# the field drifts across at phrase rate. Indices into the active scale.
DEFAULT_FIELD = [0, 5, 3, 4]  # i - vi - iv - v flavour in minor

# Curated chord progressions per scale: 0-based scale-degree roots the harmonic
# field cycles through. Every option is a known-good loop in its mode, so any
# identity pick sounds intentional rather than random.
PROGRESSIONS: Dict[str, List[List[int]]] = {
    "major": [[0, 4, 5, 3], [0, 5, 3, 4], [0, 3, 4, 0]],
    "minor": [[0, 5, 2, 6], [0, 3, 5, 4], [0, 6, 5, 4]],
    "dorian": [[0, 3, 0, 4], [0, 1, 3, 4], [0, 3, 6, 4]],
    "mixolydian": [[0, 6, 3, 0], [0, 3, 6, 4], [0, 6, 0, 4]],
    "lydian": [[0, 1, 0, 4], [0, 4, 1, 0], [0, 2, 1, 0]],
    "phrygian": [[0, 1, 0, 6], [0, 1, 3, 1], [0, 6, 1, 0]],
    "major_pentatonic": [[0, 3, 4, 0], [0, 2, 3, 0]],
    "minor_pentatonic": [[0, 3, 4, 0], [0, 2, 4, 3]],
}


def euclidean(pulses: int, steps: int) -> List[int]:
    """Bjorklund's algorithm: distribute `pulses` as evenly as possible over
    `steps`. Returns a list of 0/1 of length `steps`."""
    pulses = int(np.clip(pulses, 0, steps))
    if pulses <= 0:
        return [0] * steps
    if pulses >= steps:
        return [1] * steps
    # Downbeat-aligned even distribution: a pulse falls on step 0.
    return [1 if (i * pulses) % steps < pulses else 0 for i in range(steps)]


@dataclass
class TrackRole:
    name: str
    channel: int
    lo: int                       # lowest MIDI note
    hi: int                       # highest MIDI note
    max_density: int              # max euclidean pulses / bar
    legato: bool = False


@dataclass
class SubstrateConfig:
    tonic: int = 57               # A3
    scale: str = "minor"
    harmonic_field: List[int] = field(default_factory=lambda: list(DEFAULT_FIELD))
    steps_per_beat: int = 4
    beats_per_bar: int = 4
    bars_per_phrase: int = 4
    palette: str = "neutral"      # neutral | ambient | rhythmic (signature bias)
    roles: Dict[str, TrackRole] = field(default_factory=dict)

    @staticmethod
    def default() -> "SubstrateConfig":
        cfg = SubstrateConfig()
        cfg.roles = {
            "bass":  TrackRole("bass", channel=1, lo=33, hi=50, max_density=4),
            "chord": TrackRole("chord", channel=2, lo=52, hi=72, max_density=4,
                               legato=True),
            "lead":  TrackRole("lead", channel=3, lo=64, hi=88, max_density=8),
            "drums": TrackRole("drums", channel=10, lo=35, hi=51, max_density=16),
        }
        return cfg


class Substrate:
    """Holds harmonic state and snaps continuous values into musical ones."""

    def __init__(self, cfg: SubstrateConfig):
        self.cfg = cfg
        self._scale = SCALES[cfg.scale]
        self._phrase_pos = 0          # advances on phrase boundaries
        self._field_idx = 0

    # -- harmony -----------------------------------------------------------

    def advance_phrase(self) -> None:
        self._phrase_pos += 1
        self._field_idx = self._phrase_pos % len(self.cfg.harmonic_field)

    def set_field_index(self, i: int) -> None:
        """Drive harmony deterministically (e.g. from loop position)."""
        self._field_idx = i % len(self.cfg.harmonic_field)

    @property
    def chord_root_degree(self) -> int:
        return self.cfg.harmonic_field[self._field_idx]

    def chord_tones(self, extended: bool = False) -> List[int]:
        """Triad (1-3-5) built on the current field root, as scale degrees;
        ``extended`` adds the 7th for a richer pad voicing (still in scale)."""
        root = self.chord_root_degree
        tones = [root, root + 2, root + 4]
        if extended:
            tones.append(root + 6)
        return tones

    def scale_size(self) -> int:
        return len(self._scale)

    def degree_to_midi(self, degree: int, octave: int = 0) -> int:
        """Map an (unbounded) scale degree to a MIDI note number."""
        n = len(self._scale)
        octv = degree // n + octave
        pc = self._scale[degree % n]
        return self.cfg.tonic + 12 * octv + pc

    def fold_into_range(self, note: int, lo: int, hi: int) -> int:
        """Shift `note` by whole octaves into [lo, hi]; preserves scale membership
        (unlike a hard clip, which can land off-scale)."""
        while note < lo:
            note += 12
        while note > hi:
            note -= 12
        return int(np.clip(note, lo, hi))

    def snap(self, value01: float, role: TrackRole,
             chord_weighted: bool = True) -> int:
        """Map value in [0,1] to a MIDI note within the role's register,
        snapped to the active scale and (optionally) weighted to chord tones."""
        value01 = float(np.clip(value01, 0.0, 1.0))
        # Candidate notes in range, snapped to scale.
        candidates = []
        n = len(self._scale)
        deg = 0
        # Walk degrees spanning the register.
        start_oct = (role.lo - self.cfg.tonic) // 12 - 1
        end_oct = (role.hi - self.cfg.tonic) // 12 + 1
        for octv in range(start_oct, end_oct + 1):
            for d in range(n):
                note = self.cfg.tonic + 12 * octv + self._scale[d]
                if role.lo <= note <= role.hi:
                    candidates.append((d + octv * n, note))
        if not candidates:
            return int(np.clip(role.lo, 0, 127))
        candidates.sort(key=lambda c: c[1])
        notes = [c[1] for c in candidates]
        degs = [c[0] for c in candidates]
        # pick by position
        idx = int(round(value01 * (len(notes) - 1)))
        if chord_weighted:
            idx = self._nudge_to_chord(idx, degs)
        return int(notes[idx])

    def _nudge_to_chord(self, idx: int, degs: List[int]) -> int:
        chord = set(self.chord_tones())
        n = len(self._scale)
        # search outward for the nearest chord tone
        for off in range(0, 4):
            for s in (idx - off, idx + off):
                if 0 <= s < len(degs) and (degs[s] % n) in {c % n for c in chord}:
                    return s
        return idx

    # -- rhythm ------------------------------------------------------------

    def pattern(self, role_name: str, density01: float) -> List[int]:
        role = self.cfg.roles[role_name]
        steps = self.cfg.steps_per_beat * self.cfg.beats_per_bar
        pulses = int(round(np.clip(density01, 0, 1) * role.max_density))
        return euclidean(pulses, steps)


def biased_config(signature: dict | None) -> SubstrateConfig:
    """Bias the substrate from a person's movement signature.

    Drives palette, scale, register (tonic) and per-role density so different
    movers don't just differ numerically -- they land in different musical
    worlds. Thresholds are set against observed real-mocap scales.
    """
    cfg = SubstrateConfig.default()
    if not signature:
        return cfg
    weight = signature.get("weight_mean", signature.get("energy_mean", 0.5))
    jerk = signature.get("jerk_mean", 100.0)

    # 1) palette from how energetic / sharp the mover is
    if weight < 0.4 and jerk < 150:
        cfg.palette = "ambient"
    elif weight > 0.85 or jerk > 200:
        cfg.palette = "rhythmic"
    else:
        cfg.palette = "neutral"

    # 2) scale from palette, with a secondary split on jerk (sharp vs smooth)
    smooth = jerk < 120
    cfg.scale = {
        ("ambient", True): "major_pentatonic", ("ambient", False): "lydian",
        ("neutral", True): "minor", ("neutral", False): "dorian",
        ("rhythmic", True): "dorian", ("rhythmic", False): "phrygian",
    }[(cfg.palette, smooth)]

    # 3) register: gentle movers sit higher/brighter, energetic ones lower/darker
    cfg.tonic = {"ambient": 60, "neutral": 57, "rhythmic": 52}[cfg.palette]

    # 4) density bounds + articulation from energy
    if cfg.palette == "ambient":
        cfg.roles["drums"].max_density = 6
        cfg.roles["lead"].max_density = 4
        cfg.roles["chord"].legato = True
    elif cfg.palette == "rhythmic":
        cfg.roles["drums"].max_density = 16
        cfg.roles["lead"].max_density = 10
    else:
        cfg.roles["drums"].max_density = 10
        cfg.roles["lead"].max_density = 7
    return cfg
