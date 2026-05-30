"""Readout -- features + clock phase -> events inside the substrate.

This is where the body actually plays the instrument:
  * phase / grid crossings -> triggers (drums, bass) -- quantised, musical
  * field value -> scale index (lead / chord pitch)
  * energy -> per-step probability + density + dynamics
  * Laban Effort -> articulation, harmonic directness, FX modulation lanes
  * events (thrust / stomp / reversal) -> accents, fills, transitions
  * stillness -> a held / suspended chord, a breath -- never silence

The reactive-vs-structural blend is per track: drums & bass are quantised to the
entrained grid (musical, slightly laggy); lead accents and FX run direct off
kinematics (tight). Getting that split right is what makes it breathe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .features import Features
from .laban import Effort
from .oscillator import ClockState
from .output import MusicEvent
from .substrate import Substrate

# CC assignments (modulation lanes into your synths).
CC_DYNAMICS = 11     # expression
CC_CUTOFF = 74       # filter cutoff
CC_REVERB = 91       # reverb send
CC_DELAY = 93        # delay send
CC_PAN = 10          # stereo placement


@dataclass
class NormFeatures:
    """Personalised, range-normalised drivers in [0,1]."""
    energy: float
    openness: float
    com_height: float
    weight: float        # from Effort, passed through normaliser if desired


class Readout:
    def __init__(self, substrate: Substrate, seed: int = 0,
                 latency_compensation_s: float = 0.0):
        self.sub = substrate
        self.rng = np.random.default_rng(seed)
        self.latency = latency_compensation_s
        cfg = substrate.cfg
        self._beats_per_bar = cfg.beats_per_bar
        self._bars_per_phrase = cfg.bars_per_phrase
        self._beat_count = 0
        self._held_chord: List[int] = []      # active pad notes
        self._cc_last: Dict[int, Dict[int, int]] = {}
        self._suspended = False

    # -- public ------------------------------------------------------------

    def step(self, f: Features, eff: Effort, nf: NormFeatures,
             clk: ClockState) -> List[MusicEvent]:
        out: List[MusicEvent] = []
        still = f.events.get("freeze", 0.0)

        # Phrase / bar bookkeeping advances on each beat boundary.
        if clk.beat:
            self._beat_count += 1
            if self._beat_count % (self._beats_per_bar * self._bars_per_phrase) == 0:
                self.sub.advance_phrase()
                out += self._revoice_pad(nf, eff, still)

        # 1) Modulation lanes -- every frame, tight, off kinematics + Effort.
        out += self._modulation(f, eff, nf)

        # 2) Stillness: hold a suspended chord, a breath. Never go silent.
        if still > 0.6 or not f.present:
            out += self._enter_stillness(nf, eff)
            return out
        if self._suspended:
            out += self._exit_stillness()

        # 3) Quantised lanes (drums, bass) -- fire on grid steps.
        if clk.step_advanced and clk.locked:
            out += self._grid_lane(f, eff, nf, clk)

        # 4) Reactive lanes (lead, accents) -- direct off kinematics / events.
        out += self._reactive_lane(f, eff, nf, clk)

        # 5) Ensure a pad exists once we're playing.
        if not self._held_chord and clk.locked:
            out += self._revoice_pad(nf, eff, still)

        return out

    # -- lanes -------------------------------------------------------------

    def _grid_lane(self, f, eff, nf, clk) -> List[MusicEvent]:
        out: List[MusicEvent] = []
        cfg = self.sub.cfg
        steps = cfg.steps_per_beat * cfg.beats_per_bar
        step = clk.step % cfg.steps_per_beat + (
            (self._beat_count % cfg.beats_per_bar) * cfg.steps_per_beat)
        step %= steps

        # Drums: kick pattern density from energy + Weight; hats from energy.
        kick = self.sub.pattern("drums", 0.25 + 0.5 * nf.energy)
        hat = self.sub.pattern("drums", 0.4 + 0.6 * nf.energy)
        if kick[step] and self.rng.random() < 0.7 + 0.3 * eff.weight:
            vel = int(np.clip(60 + 60 * eff.weight, 1, 127))
            out.append(MusicEvent("note_on", 10, 36, vel, dur=0.08, tag="drums.kick"))
        if hat[step] and self.rng.random() < 0.4 + 0.5 * nf.energy:
            vel = int(np.clip(40 + 50 * eff.time, 1, 127))
            out.append(MusicEvent("note_on", 10, 42, vel, dur=0.05, tag="drums.hat"))

        # Bass: root of the field on strong steps, density from energy.
        bass = self.sub.pattern("bass", 0.3 + 0.5 * nf.energy)
        if bass[step]:
            role = cfg.roles["bass"]
            note = self.sub.degree_to_midi(self.sub.chord_root_degree, octave=-1)
            note = int(np.clip(note, role.lo, role.hi))
            vel = int(np.clip(55 + 55 * eff.weight, 1, 127))
            dur = 60.0 / max(clk.bpm, 1) * (0.9 if eff.flow > 0.5 else 0.4)
            out.append(MusicEvent("note_on", role.channel, note, vel,
                                  dur=dur, tag="bass"))
        return out

    def _reactive_lane(self, f, eff, nf, clk) -> List[MusicEvent]:
        out: List[MusicEvent] = []
        role = self.sub.cfg.roles["lead"]

        # Probability of a lead note this beat-step, gated by energy.
        # Space (directness) raises melodic activity; Flow lengthens notes.
        fire = False
        if clk.step_advanced:
            p = 0.15 + 0.6 * nf.energy * (0.5 + 0.5 * eff.space)
            fire = self.rng.random() < p
        # Events always punch through (tight, off kinematics).
        thrust = f.events.get("thrust", 0.0)
        reversal = f.events.get("reversal", 0.0)
        accent = max(thrust, reversal)
        if accent > 0.3:
            fire = True

        if fire:
            # Field value: openness + height pick the scale index / register.
            field_val = float(np.clip(0.5 * nf.openness + 0.5 * nf.com_height, 0, 1))
            note = self.sub.snap(field_val, role, chord_weighted=eff.space > 0.4)
            vel = int(np.clip(50 + 50 * eff.weight + 40 * accent, 1, 127))
            # Flow -> articulation: free = legato (long), bound = staccato (short).
            beat_s = 60.0 / max(clk.bpm, 1)
            dur = beat_s * (0.15 + 0.7 * eff.flow)
            out.append(MusicEvent("note_on", role.channel, note, vel,
                                  dur=dur, tag="lead"))

        # Stomp -> extra low percussion hit (fill).
        stomp = f.events.get("stomp", 0.0)
        if stomp > 0.4:
            out.append(MusicEvent("note_on", 10, 41, int(60 + 60 * stomp),
                                  dur=0.1, tag="drums.tom"))
        return out

    def _modulation(self, f, eff, nf) -> List[MusicEvent]:
        out: List[MusicEvent] = []
        # Dynamics from energy + Weight; cutoff from energy (brightness).
        out += self._cc(CC_DYNAMICS, 0, 30 + 90 * nf.energy)
        out += self._cc(CC_CUTOFF, 0, 30 + 95 * (0.4 * nf.energy + 0.6 * eff.weight))
        # Flow (free) -> more reverb; Space (indirect) -> more delay.
        out += self._cc(CC_REVERB, 0, 20 + 90 * (1.0 - 0.5 * abs(eff.flow - 1.0)) * eff.flow)
        out += self._cc(CC_DELAY, 0, 15 + 90 * (1.0 - eff.space))
        # Stereo placement from L/R asymmetry (call/response readout hook).
        pan = int(np.clip(64 + 60 * f.asymmetry_lr, 0, 127))
        out += self._cc(CC_PAN, 3, pan)
        return out

    # -- pad / stillness ---------------------------------------------------

    def _revoice_pad(self, nf, eff, still) -> List[MusicEvent]:
        out: List[MusicEvent] = []
        out += self._release_pad()
        role = self.sub.cfg.roles["chord"]
        tones = self.sub.chord_tones()
        # Openness -> voicing spread (register span of the chord).
        spread = 0.3 + 0.7 * nf.openness
        base_oct = int(round(nf.com_height * 1.0))
        notes = []
        for k, deg in enumerate(tones):
            octave = base_oct + int(k * spread)
            note = self.sub.degree_to_midi(deg, octave=octave)
            note = int(np.clip(note, role.lo, role.hi))
            notes.append(note)
        vel = int(np.clip(35 + 45 * eff.weight, 1, 100))
        for note in sorted(set(notes)):
            out.append(MusicEvent("note_on", role.channel, note, vel, tag="chord"))
        self._held_chord = sorted(set(notes))
        return out

    def _release_pad(self) -> List[MusicEvent]:
        role = self.sub.cfg.roles["chord"]
        out = [MusicEvent("note_off", role.channel, n, 0, tag="chord")
               for n in self._held_chord]
        self._held_chord = []
        return out

    def _enter_stillness(self, nf, eff) -> List[MusicEvent]:
        out: List[MusicEvent] = []
        if not self._suspended:
            # Suspended/breathing chord: revoice with an added 2nd for openness.
            out += self._release_pad()
            role = self.sub.cfg.roles["chord"]
            root = self.sub.chord_root_degree
            sus = [root, root + 1, root + 4]  # sus2-ish colour
            notes = sorted({int(np.clip(self.sub.degree_to_midi(d, octave=k),
                                        role.lo, role.hi))
                            for k, d in enumerate(sus)})
            for n in notes:
                out.append(MusicEvent("note_on", role.channel, n, 45, tag="chord.sus"))
            self._held_chord = notes
            self._suspended = True
        # A slow breath on reverb so stillness is alive, not dead.
        out += self._cc(CC_REVERB, 0, 90, force=True)
        return out

    def _exit_stillness(self) -> List[MusicEvent]:
        self._suspended = False
        return []

    # -- helpers -----------------------------------------------------------

    def _cc(self, num: int, channel: int, value: float,
            force: bool = False) -> List[MusicEvent]:
        v = int(np.clip(round(value), 0, 127))
        last = self._cc_last.setdefault(channel, {})
        if not force and abs(last.get(num, -999) - v) < 2:
            return []
        last[num] = v
        return [MusicEvent("cc", channel, num, v, tag=f"cc.{num}")]
