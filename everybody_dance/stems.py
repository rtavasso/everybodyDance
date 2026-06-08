"""Multi-stem looper rack: rhythm + pitch + timbre + FX -> audio / MIDI events.

Synth and FX are *injected* callables (SynthFn / FxFn protocols) so this
module can be tested in isolation without everybody_dance.synth_engine or
everybody_dance.fx.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence
import numpy as np

from everybody_dance.studio_types import (
    FxConfig, Note, PitchConfig, RhythmConfig, StemConfig, StemState,
    TimbreConfig, SynthFn, FxFn, SR,
)
from everybody_dance.substrate import SCALES
from everybody_dance.output import MusicEvent

# ---------------------------------------------------------------------------
# GM drum map (channel 10).
# ---------------------------------------------------------------------------

_DRUM_HEIGHTS = [
    (0.00, 0.40, 36),   # kick
    (0.40, 0.70, 38),   # snare
    (0.70, 1.01, 42),   # closed hi-hat
]


def _drum_pitch(height01: float) -> int:
    for lo, hi, note in _DRUM_HEIGHTS:
        if lo <= height01 < hi:
            return note
    return 42   # fallback: hat


# ---------------------------------------------------------------------------
# Euclidean (Bjorklund) rhythm.
# ---------------------------------------------------------------------------

def euclid(steps: int, pulses: int, rotation: int = 0) -> list[bool]:
    """Bjorklund algorithm: distribute `pulses` hits across `steps` as evenly
    as possible.  Returns a list[bool] of length `steps`."""
    pulses = int(np.clip(pulses, 0, steps))
    if pulses <= 0:
        return [False] * steps
    if pulses >= steps:
        return [True] * steps
    # Standard Euclidean sequence: 1 if (i*pulses) % steps < pulses
    pattern = [(i * pulses) % steps < pulses for i in range(steps)]
    # Apply rotation (positive = shift right)
    r = rotation % steps
    return pattern[-r:] + pattern[:-r] if r else pattern


# ---------------------------------------------------------------------------
# Pitch snapping.
# ---------------------------------------------------------------------------

def snap_pitch(height01: float, pitch: PitchConfig) -> int:
    """Map 0..1 body height to a MIDI note within pitch.register.

    If pitch.quantize, snap to the nearest scale degree (SCALES[pitch.scale]
    rooted at pitch.tonic).  mode="fixed" ignores height and uses
    pitch.fixed_degree as an offset into the scale from the tonic.
    """
    lo, hi = pitch.register
    intervals = SCALES.get(pitch.scale, SCALES["minor"])

    if pitch.mode == "fixed":
        # Build scale note from tonic + fixed_degree offset
        octave, deg = divmod(pitch.fixed_degree, len(intervals))
        midi = pitch.tonic + octave * 12 + intervals[deg]
        return int(np.clip(midi, lo, hi))

    # Map height01 to raw MIDI note inside register
    raw = lo + height01 * (hi - lo)
    raw = float(np.clip(raw, lo, hi))

    if not pitch.quantize:
        return int(round(raw))

    # Build all scale notes that span [lo-6, hi+6] (enough to always snap)
    root = pitch.tonic % 12
    candidates: list[int] = []
    for octave in range(-2, 12):
        for interval in intervals:
            note = root + interval + octave * 12
            if lo - 6 <= note <= hi + 6:
                candidates.append(note)
    if not candidates:
        return int(round(raw))

    # Nearest candidate, clamped to register
    arr = np.array(candidates, dtype=np.float32)
    nearest = int(candidates[int(np.argmin(np.abs(arr - raw)))])
    return int(np.clip(nearest, lo, hi))


# ---------------------------------------------------------------------------
# StemRack.
# ---------------------------------------------------------------------------

class StemRack:
    """Ties rhythm, pitch, timbre, and FX together across multiple stems."""

    def __init__(self, configs: list[StemConfig]) -> None:
        self.states: list[StemState] = [
            StemState(
                config=cfg,
                loop=[None] * cfg.rhythm.total_steps,
            )
            for cfg in configs
        ]

    # ------------------------------------------------------------------
    # Authoring helpers.
    # ------------------------------------------------------------------

    def author_euclid(self, i: int) -> None:
        """Fill stem i's loop with notes derived from its Euclidean rhythm.

        Heights are spread evenly across the pitch register (simple upward
        arpeggio) so the melody has some variety.  Drums use height->GM map.
        """
        st = self.states[i]
        cfg = st.config
        r = cfg.rhythm
        pattern = euclid(r.total_steps, r.pulses, r.rotation)
        n_hits = sum(pattern)
        st.loop = [None] * r.total_steps

        hit_idx = 0
        for step, active in enumerate(pattern):
            if not active:
                continue
            # Spread hits evenly 0..1 for arpeggio effect
            height = hit_idx / max(n_hits - 1, 1) if n_hits > 1 else 0.5
            hit_idx += 1

            if cfg.channel == 10:
                pitch_val = _drum_pitch(height)
            else:
                pitch_val = snap_pitch(height, cfg.pitch)

            note = Note(
                step=step,
                pitch=pitch_val,
                vel=90,
                dur_steps=max(r.gate, 0.1),
                height01=height,
                tag="euclid",
            )
            st.loop[step] = note

    def record_onset(self, i: int, step: int, vel: int, height01: float) -> None:
        """Place (or overdub) a Note on stem i at `step`.

        Pitch is resolved via snap_pitch from height01; drums (channel 10)
        use GM mapping.
        """
        st = self.states[i]
        cfg = st.config
        height01 = float(np.clip(height01, 0.0, 1.0))
        step = int(step) % cfg.rhythm.total_steps

        if cfg.channel == 10:
            pitch_val = _drum_pitch(height01)
        else:
            pitch_val = snap_pitch(height01, cfg.pitch)

        note = Note(
            step=step,
            pitch=pitch_val,
            vel=int(np.clip(vel, 1, 127)),
            dur_steps=max(cfg.rhythm.gate, 0.1),
            height01=height01,
        )
        st.loop[step] = note

    def clear(self, i: int) -> None:
        """Erase all notes from stem i's loop."""
        st = self.states[i]
        st.loop = [None] * st.config.rhythm.total_steps

    def set_muted(self, i: int, muted: bool) -> None:
        self.states[i].muted = muted

    def set_solo(self, i: int, solo: bool) -> None:
        self.states[i].solo = solo

    # ------------------------------------------------------------------
    # Live parameter mapping.
    # ------------------------------------------------------------------

    def apply_mapping(self, i: int, signals: dict[str, float]) -> None:
        """For each binding `signal -> dotted.path` in the stem's Mapping,
        write the signal value (0..1) into the target parameter.

        Supported targets: fx.<field>, rhythm.<field>, timbre.<field>,
        pitch.<field>.  Values are clamped to [0, 1] (or int where needed).
        """
        st = self.states[i]
        cfg = st.config
        for signal, path in cfg.mapping.bindings.items():
            if signal not in signals:
                continue
            val = float(np.clip(signals[signal], 0.0, 1.0))
            parts = path.split(".", 1)
            if len(parts) != 2:
                continue
            section, attr = parts
            target = {
                "fx": cfg.fx,
                "rhythm": cfg.rhythm,
                "timbre": cfg.timbre,
                "pitch": cfg.pitch,
            }.get(section)
            if target is None or not hasattr(target, attr):
                continue
            # Preserve the type of the existing value (int vs float)
            existing = getattr(target, attr)
            if isinstance(existing, bool):
                setattr(target, attr, val >= 0.5)
            elif isinstance(existing, int):
                setattr(target, attr, int(round(val)))
            else:
                setattr(target, attr, val)

    # ------------------------------------------------------------------
    # Timing helpers.
    # ------------------------------------------------------------------

    def step_dur(self, bpm: float, stem_i: int = 0) -> float:
        """Return duration in seconds of one grid step for stem i.

        One bar contains `rhythm.steps` grid steps and spans 4 beats:
            step_dur = (60 / bpm) * (4 / steps)

        Swing is applied externally per-step in render(); this returns the
        base (un-swung) step duration.
        """
        steps = self.states[stem_i].config.rhythm.steps
        return (60.0 / bpm) * (4.0 / steps)

    def _step_time(self, step: int, base_dur: float, swing: float) -> float:
        """Absolute time of `step` in seconds with swing on odd steps.

        Odd steps are pushed forward by `swing * base_dur * 0.5`.
        """
        t = step * base_dur
        if step % 2 == 1:
            t += swing * base_dur * 0.5
        return t

    # ------------------------------------------------------------------
    # Render.
    # ------------------------------------------------------------------

    def render(
        self,
        sr: int,
        bpm: float,
        synth_fn: SynthFn,
        fx_fn: FxFn,
        cycles: int = 2,
    ) -> np.ndarray:
        """Render all audible stems to a mono float32 mix.

        Steps:
        1. Determine which stems are audible (mute/solo logic).
        2. For each audible stem, render `cycles` repetitions of its loop by
           calling synth_fn for each Note; collect into a stem buffer.
        3. Pass the stem buffer through fx_fn (may lengthen it via tails).
        4. Sum stem buffers into the mix; tanh-normalise to ~0.95 peak.
        """
        any_solo = any(s.solo for s in self.states)

        def _audible(st: StemState) -> bool:
            if st.muted:
                return False
            if any_solo and not st.solo:
                return False
            return True

        # Compute the full duration of `cycles` loops for the longest stem.
        max_loop_s = 0.0
        for st in self.states:
            r = st.config.rhythm
            base = (60.0 / bpm) * (4.0 / r.steps)
            loop_s = r.total_steps * base  # un-swung estimate (close enough)
            max_loop_s = max(max_loop_s, loop_s * cycles)

        total_samples = int(np.ceil(max_loop_s * sr)) + sr  # +1 s headroom for FX tails

        mix = np.zeros(total_samples, dtype=np.float32)

        for st in self.states:
            if not _audible(st):
                continue

            r = st.config.rhythm
            base_dur = (60.0 / bpm) * (4.0 / r.steps)
            loop_s = r.total_steps * base_dur

            stem_buf = np.zeros(total_samples, dtype=np.float32)

            for cycle in range(cycles):
                cycle_offset_s = cycle * loop_s
                for note in st.loop:
                    if note is None:
                        continue
                    t_s = cycle_offset_s + self._step_time(note.step, base_dur, r.swing)
                    dur_s = note.dur_steps * base_dur * r.gate
                    dur_s = max(dur_s, 0.001)
                    note_buf = synth_fn(note.pitch, dur_s, note.vel, st.config.timbre, sr)
                    note_buf = np.asarray(note_buf, dtype=np.float32)
                    start = int(round(t_s * sr))
                    end = start + len(note_buf)
                    if end > len(stem_buf):
                        end = len(stem_buf)
                        note_buf = note_buf[: end - start]
                    if start < len(stem_buf) and end > start:
                        stem_buf[start:end] += note_buf

            # Apply FX (may lengthen the buffer).
            processed = fx_fn(stem_buf, st.config.fx, sr, bpm)
            processed = np.asarray(processed, dtype=np.float32)
            # Accumulate into mix (pad mix if FX returned a longer buffer).
            if len(processed) > len(mix):
                mix = np.concatenate([mix, np.zeros(len(processed) - len(mix), dtype=np.float32)])
            mix[: len(processed)] += processed

        # Tanh soft-clip normalise to ~0.95 peak.
        peak = np.max(np.abs(mix))
        if peak > 0:
            mix = np.tanh(mix / peak) * 0.95

        return mix

    # ------------------------------------------------------------------
    # MIDI event list.
    # ------------------------------------------------------------------

    def to_events(self, bpm: float, cycles: int = 2) -> list[MusicEvent]:
        """Return a flat list of MusicEvent (note_on + note_off pairs).

        Timestamps are in seconds from the start of the render.
        """
        events: list[MusicEvent] = []

        for st in self.states:
            r = st.config.rhythm
            base_dur = (60.0 / bpm) * (4.0 / r.steps)
            loop_s = r.total_steps * base_dur
            ch = st.config.channel

            for cycle in range(cycles):
                cycle_offset_s = cycle * loop_s
                for note in st.loop:
                    if note is None:
                        continue
                    t_on = cycle_offset_s + self._step_time(note.step, base_dur, r.swing)
                    dur_s = note.dur_steps * base_dur * r.gate
                    dur_s = max(dur_s, 0.001)
                    t_off = t_on + dur_s

                    events.append(MusicEvent(
                        kind="note_on",
                        channel=ch,
                        a=note.pitch,
                        b=note.vel,
                        t=t_on,
                        dur=dur_s,
                        tag=st.config.name,
                    ))
                    events.append(MusicEvent(
                        kind="note_off",
                        channel=ch,
                        a=note.pitch,
                        b=0,
                        t=t_off,
                        tag=st.config.name,
                    ))

        events.sort(key=lambda e: e.t)
        return events
