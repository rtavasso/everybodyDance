"""Shared type contract for the everybodyDance *studio* — the polished,
customizable, multi-stem rhythm/pitch/timbre/effects instrument with loopers and
a Just-Dance-inspired UI.

This module is the integration backbone: the synth engine, FX chain, stem/looper
rack, and UI are each built against these dataclasses + the function Protocols at
the bottom, so they compose cleanly. Keep it dependency-light (numpy + typing).
Nothing here does DSP or drawing; it only defines the shapes everyone agrees on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Per-stem configuration -- the "highly customizable" surface.
# ---------------------------------------------------------------------------

@dataclass
class RhythmConfig:
    """How a stem's onsets are placed on the loop grid."""
    mode: str = "body"          # "body" (movement hits author onsets) | "euclid" | "manual"
    steps: int = 16             # grid resolution for one loop bar
    bars: int = 2               # loop length in bars
    pulses: int = 4             # euclid: number of hits
    rotation: int = 0           # euclid: rotate the pattern
    swing: float = 0.0          # 0..1 swing on off-beats
    gate: float = 0.9           # note length as a fraction of the step
    density: float = 0.5        # body-mode hit sensitivity / euclid fill bias

    @property
    def total_steps(self) -> int:
        return self.steps * self.bars


@dataclass
class PitchConfig:
    """How a stem's notes get their pitch."""
    mode: str = "height"        # "height" (COM) | "limb" (hand height) | "fixed" | "arp"
    scale: str = "minor"
    tonic: int = 57             # MIDI note of the tonic
    register: Tuple[int, int] = (48, 72)   # (low, high) MIDI clamp
    quantize: bool = True       # snap to scale
    arp: str = "up"             # arp mode: up|down|updown|random
    glide: float = 0.0          # 0..1 portamento amount
    fixed_degree: int = 0       # for mode="fixed": scale degree offset


@dataclass
class TimbreConfig:
    """A stem's sound. Consumed by the synth engine."""
    instrument: str = "pluck"   # pluck|pad|fm|saw|square|sine|bass|bell|kick|snare|hat|noise
    attack: float = 0.005
    decay: float = 0.20
    sustain: float = 0.6        # 0..1
    release: float = 0.20
    cutoff: float = 0.85        # 0..1 normalized lowpass cutoff
    resonance: float = 0.10     # 0..1
    harmonics: int = 4          # additive richness where relevant
    detune: float = 0.0         # 0..1 unison detune
    fm_ratio: float = 2.0       # for instrument="fm"
    fm_index: float = 3.0
    gain: float = 0.8


@dataclass
class FxConfig:
    """Per-stem (and master) effect chain. Consumed by the FX engine."""
    drive: float = 0.0          # 0..1 saturation
    bitcrush: float = 0.0       # 0..1 (0 = off) sample/bit reduction
    filter_cutoff: float = 1.0  # 0..1 post-filter (1 = open)
    filter_res: float = 0.0     # 0..1
    delay_send: float = 0.0     # 0..1 wet
    delay_time: float = 0.375   # delay time in beats
    delay_feedback: float = 0.35
    reverb_send: float = 0.0    # 0..1 wet
    reverb_size: float = 0.5    # 0..1 room size


@dataclass
class Mapping:
    """Live customization: body signal name -> dotted param path.
    e.g. {"openness": "fx.filter_cutoff", "energy": "rhythm.density",
          "limb_energy": "fx.delay_send"}. Values are read 0..1 and written
          to the target param (consumers clamp/scale as needed)."""
    bindings: Dict[str, str] = field(default_factory=dict)


@dataclass
class StemConfig:
    name: str
    channel: int                       # MIDI-ish channel (10 = drums)
    rhythm: RhythmConfig = field(default_factory=RhythmConfig)
    pitch: PitchConfig = field(default_factory=PitchConfig)
    timbre: TimbreConfig = field(default_factory=TimbreConfig)
    fx: FxConfig = field(default_factory=FxConfig)
    mapping: Mapping = field(default_factory=Mapping)
    color: Tuple[int, int, int] = (200, 200, 200)   # BGR for the UI


# ---------------------------------------------------------------------------
# Runtime note + loop model.
# ---------------------------------------------------------------------------

@dataclass
class Note:
    step: int                   # position in the loop grid (0..total_steps-1)
    pitch: int                  # MIDI note (for drums: GM piece number)
    vel: int = 100              # 1..127
    dur_steps: float = 1.0      # length in grid steps
    height01: float = 0.5       # body height captured at authoring (for re-voicing)
    tag: str = ""


@dataclass
class StemState:
    config: StemConfig
    loop: List[Optional[Note]]              # length == config.rhythm.total_steps
    recording: bool = False
    overdub: bool = False
    muted: bool = False
    solo: bool = False
    ghost: Optional[List[np.ndarray]] = None   # one loop of skeletons for the UI

    @property
    def n_onsets(self) -> int:
        return sum(s is not None for s in self.loop)


# ---------------------------------------------------------------------------
# UI / game state (consumed by the Just-Dance-style compositor).
# ---------------------------------------------------------------------------

@dataclass
class MovePrompt:
    """A Just-Dance 'do this move now' cue scrolling toward the hit line."""
    name: str
    due_t: float                # wall/loop time the move should land
    lead_s: float = 2.0         # how early it appears
    hit: Optional[bool] = None  # None=pending, True=nailed, False=missed


@dataclass
class StudioUIState:
    stems: List[StemState]
    active_stem: int = 0
    phase: str = "perform"      # idle|record_rhythm|record_pitch|perform
    playhead: int = 0           # current grid step within the loop
    bpm: float = 100.0
    pitch01: float = 0.5        # current body-height meter
    flashes: List = field(default_factory=list)        # effects.Flash list
    prompts: List[MovePrompt] = field(default_factory=list)
    score: int = 0
    combo: int = 0
    rating: str = ""            # "PERFECT"/"GOOD"/"MISS" feedback flash
    meters: Dict[str, float] = field(default_factory=dict)  # named 0..1 viz values
    live_xyz: Optional[np.ndarray] = None
    present: bool = True
    bar: int = 0
    bars_left: int = 0


# ---------------------------------------------------------------------------
# Function contracts (Protocols) -- implemented by the engine modules.
# ---------------------------------------------------------------------------

class SynthFn(Protocol):
    """Render one note to a mono float32 buffer (gain per timbre.gain applied,
    NOT globally normalised). Percussion instruments ignore `pitch`."""
    def __call__(self, pitch: int, dur_s: float, vel: int,
                 timbre: TimbreConfig, sr: int) -> np.ndarray: ...


class FxFn(Protocol):
    """Apply a stem's FX chain to a mono buffer. May return a longer buffer
    (delay/reverb tails). Deterministic."""
    def __call__(self, buf: np.ndarray, fx: FxConfig, sr: int,
                 bpm: float) -> np.ndarray: ...


SR = 22050


# ---------------------------------------------------------------------------
# Convenient defaults: a tasteful 4-stem starter kit.
# ---------------------------------------------------------------------------

def default_stem_configs() -> List[StemConfig]:
    return [
        StemConfig("drums", 10, color=(80, 120, 255),
                   rhythm=RhythmConfig(mode="body", steps=16, bars=2, gate=0.4),
                   timbre=TimbreConfig(instrument="kick", decay=0.18, gain=0.95),
                   pitch=PitchConfig(mode="fixed")),
        StemConfig("bass", 1, color=(80, 220, 120),
                   rhythm=RhythmConfig(mode="body", steps=16, bars=2, gate=0.8),
                   timbre=TimbreConfig(instrument="bass", cutoff=0.5, gain=0.85),
                   pitch=PitchConfig(mode="height", register=(36, 52))),
        StemConfig("keys", 2, color=(240, 180, 70),
                   rhythm=RhythmConfig(mode="euclid", steps=16, bars=2, pulses=6, gate=1.0),
                   timbre=TimbreConfig(instrument="pad", attack=0.08, release=0.4, gain=0.4),
                   pitch=PitchConfig(mode="height", register=(55, 79)),
                   fx=FxConfig(reverb_send=0.3, reverb_size=0.6)),
        StemConfig("lead", 3, color=(200, 100, 240),
                   rhythm=RhythmConfig(mode="body", steps=16, bars=2, gate=0.5),
                   timbre=TimbreConfig(instrument="pluck", decay=0.25, gain=0.6),
                   pitch=PitchConfig(mode="height", register=(60, 84)),
                   fx=FxConfig(delay_send=0.3, delay_time=0.375, delay_feedback=0.4)),
    ]
