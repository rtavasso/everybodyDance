"""Per-stem mechanics + step-quantised loopers for the Studio.

A :class:`Stem` owns one musical lane (drums/bass/keys/lead/texture): its
TrackRole, its current StemParams (from mapping), a per-step loop buffer
(reused from the looper.py pattern), and the transient toggles a gesture can
flip (mute/solo/record/timbre morph). The :class:`StemRack` wires the fixed set
of stems to substrate roles -- drums/bass/lead come from ``biased_config`` so a
person's signature still shapes register/density; ``keys`` borrows the chord
role; ``texture`` is a wide, high, legato, low-density role added on top.

Generation is body-authored but deterministic: euclidean placement from the
mapped density, pitch snapped to scale via the Substrate, velocity from
energy/weight, duration from flow/role. Stillness gating lives in the Studio;
here a stem just turns a request into in-scale MusicEvents.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .mapping import StemParams
from .output import MusicEvent
from .substrate import Substrate, TrackRole, euclidean

# Stem -> MIDI channel (drums on 10 GM-style). The order is the stem order.
CHANNELS: Dict[str, int] = {"drums": 10, "bass": 1, "keys": 2, "lead": 3,
                            "texture": 4}
STEM_ORDER: List[str] = ["drums", "bass", "keys", "lead", "texture"]

# BGR lane colours for the stage (match the effects.STYLE palette in spirit).
STEM_COLOR: Dict[str, tuple] = {
    "drums": (80, 120, 255), "bass": (255, 150, 80), "keys": (90, 240, 255),
    "lead": (160, 220, 255), "texture": (200, 160, 255)}

# Drum pitch -> piece, GM-ish (kick/hat/snare/tom). Picked by pitch position.
DRUM_PIECE = {"kick": 36, "hat": 42, "snare": 38, "tom": 41}

# Per-role note durations (legato lanes ring longer).
DUR = {"drums": 0.06, "bass": 0.45, "keys": 1.4, "lead": 0.3, "texture": 1.8}


def texture_role(channel: int = 4) -> TrackRole:
    """A wide, high-ish, legato lane that sits quietly under everything."""
    return TrackRole("texture", channel=channel, lo=60, hi=84, max_density=3,
                     legato=True)


@dataclass
class Stem:
    """One musical lane: role + live params + its step-quantised looper."""
    name: str
    role: TrackRole
    channel: int
    color: tuple
    loop_steps: int
    params: StemParams = field(default_factory=lambda: StemParams("?"))
    timbre: str = ""                          # current preset (timbre_morph)
    muted: bool = False
    soloed: bool = False
    recording: bool = False                   # capturing into the buffer
    looping: bool = False                     # a locked loop is playing
    level: float = 0.0                        # recent note energy 0..1 (VU)
    active: bool = False                      # produced a note this step
    buffer: List[List[MusicEvent]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.buffer:
            self.buffer = [[] for _ in range(self.loop_steps)]

    # -- looper mechanics (the looper.py step-quantised buffer approach) ----

    def start_record(self) -> None:
        """First fire: arm + clear; second fire: lock the captured loop."""
        if not self.recording:
            self.recording = True
            self.looping = False
            self.buffer = [[] for _ in range(self.loop_steps)]
        else:
            self.recording = False
            self.looping = any(self.buffer)

    def clear_loop(self) -> None:
        self.recording = False
        self.looping = False
        self.buffer = [[] for _ in range(self.loop_steps)]

    def toggle_loop(self) -> None:
        """Mute/unmute a locked loop (no-op while still recording)."""
        if self.looping or any(self.buffer):
            self.muted = not self.muted

    def capture(self, step: int, events: List[MusicEvent]) -> None:
        if self.recording:
            self.buffer[step] = [copy.copy(e) for e in events]

    def replay(self, step: int) -> List[MusicEvent]:
        if not self.looping:
            return []
        return [copy.copy(e) for e in self.buffer[step]]

    def loop_onsets(self) -> List[bool]:
        return [any(e.kind == "note_on" for e in slot) for slot in self.buffer]

    # -- state for the UI snapshot -----------------------------------------

    def decay_level(self, dt: float) -> None:
        self.level = float(max(0.0, self.level - dt * 1.8))


class StemRack:
    """The fixed set of stems, wired to substrate roles.

    drums/bass/lead reuse the per-person ``biased_config`` roles; ``keys`` maps
    to the substrate ``chord`` role; ``texture`` is added wide/high/legato.
    """

    def __init__(self, sub: Substrate, loop_steps: int,
                 timbres: Optional[Dict[str, str]] = None):
        self.sub = sub
        self.loop_steps = loop_steps
        timbres = timbres or {}
        # Make sure the substrate carries a texture role for snap()/pattern().
        if "texture" not in sub.cfg.roles:
            sub.cfg.roles["texture"] = texture_role(CHANNELS["texture"])
        role_for = {"drums": "drums", "bass": "bass", "keys": "chord",
                    "lead": "lead", "texture": "texture"}
        self.stems: Dict[str, Stem] = {}
        for name in STEM_ORDER:
            role = sub.cfg.roles[role_for[name]]
            self.stems[name] = Stem(
                name=name, role=role, channel=CHANNELS[name],
                color=STEM_COLOR[name], loop_steps=loop_steps,
                timbre=timbres.get(name, ""))

    def __iter__(self):
        return (self.stems[n] for n in STEM_ORDER)

    def get(self, name: str) -> Optional[Stem]:
        return self.stems.get(name)

    def any_soloed(self) -> bool:
        return any(s.soloed for s in self.stems.values())

    def audible(self, stem: Stem) -> bool:
        """A stem sounds unless muted, or unless something else is soloed."""
        if self.any_soloed():
            return stem.soloed
        return not stem.muted


# -- generation ------------------------------------------------------------

def _velocity(base: float, energy: float, weight: float, accent: float = 0.0
              ) -> int:
    return int(np.clip(base + 45 * (0.5 * energy + 0.5 * weight) + 35 * accent,
                       1, 127))


def generate(stem: Stem, sub: Substrate, step: int, steps_per_bar: int,
             sig: dict, beat_s: float, density: float,
             vel_bias: float = 0.0) -> List[MusicEvent]:
    """Body -> this stem's note_on events for one grid step (deterministic).

    ``density`` is the (already modulated) 0..1 placement density; ``sig`` is
    the studio signal dict (mapping keys); ``vel_bias`` is an additive velocity
    push (streak / combo payoff). No RNG -- euclidean placement plus
    scale-snapped pitch keep it replay-identical.
    """
    p = stem.params
    bar_step = step % steps_per_bar
    energy = float(sig.get("energy", 0.0))
    weight = float(sig.get("weight", 0.0))
    flow = float(sig.get("flow", 0.5))
    out: List[MusicEvent] = []

    if stem.name == "drums":
        kick = sub.pattern("drums", density)
        hat = sub.pattern("drums", min(1.0, 0.2 + density))
        if kick[bar_step]:
            out.append(MusicEvent("note_on", stem.channel, DRUM_PIECE["kick"],
                                  _velocity(55 + vel_bias, energy, weight),
                                  dur=DUR["drums"], tag="drums"))
        if hat[bar_step] and bar_step % 2 == 0:
            out.append(MusicEvent("note_on", stem.channel, DRUM_PIECE["hat"],
                                  _velocity(35 + vel_bias, energy, weight),
                                  dur=DUR["drums"], tag="drums"))
        return out

    if stem.name == "texture":
        # POLYRHYTHM: a 3- or 5-pulse euclidean over the bar (against the 4/4
        # everyone else plays), rotated per bar so it phases -- the ambient
        # lane orbits the beat instead of doubling it.
        pulses = 0 if density <= 0.0 else (3 if density < 0.66 else 5)
        base = euclidean(pulses, steps_per_bar)
        rot = ((step // steps_per_bar) % 4) * 2
        pat = base[rot:] + base[:rot]
    else:
        pat = sub.pattern(stem.role.name, density)
    if not pat[bar_step]:
        return out

    field_val = float(np.clip(p.pitch, 0.0, 1.0))
    note = sub.snap(field_val, stem.role, chord_weighted=stem.name != "lead")
    # Apply the stem's register octave (then fold back into the role range).
    note = sub.fold_into_range(note + 12 * int(p.register_octave),
                               stem.role.lo, stem.role.hi)
    dur = DUR[stem.name] * (0.6 + 0.8 * flow if stem.name == "lead" else 1.0)
    out.append(MusicEvent("note_on", stem.channel, note,
                          _velocity(48 + vel_bias, energy, weight),
                          dur=beat_s_to_dur(dur, beat_s), tag=stem.name))
    return out


def beat_s_to_dur(dur_units: float, beat_s: float) -> float:
    """Bass/lead durations scale with tempo; pads/keys use absolute seconds."""
    return float(dur_units)


def revoice_pad(stem: Stem, sub: Substrate, sig: dict, beat_s: float,
                dur_s: float = None) -> List[MusicEvent]:
    """One soft sustained chord for keys/texture -- used as the single
    stillness-onset allowance and as the per-bar pad revoice. ``dur_s``
    overrides the legacy fixed duration with a tempo-aware one."""
    openness = float(np.clip(sig.get("openness", 0.5), 0, 1))
    weight = float(np.clip(sig.get("weight", 0.3), 0, 1))
    spread = 0.3 + 0.7 * openness
    base_oct = int(round(float(np.clip(sig.get("com_height", 0.5), 0, 1))))
    # an open body earns the richer 7th-chord voicing
    tones = sub.chord_tones(extended=openness > 0.6)
    notes = sorted({sub.fold_into_range(
        sub.degree_to_midi(d, octave=base_oct + int(k * spread)),
        stem.role.lo, stem.role.hi) for k, d in enumerate(tones)})
    vel = int(np.clip(28 + 40 * weight, 1, 90))
    return [MusicEvent("note_on", stem.channel, n, vel,
                       dur=dur_s if dur_s is not None else DUR[stem.name],
                       tag=stem.name) for n in notes]
