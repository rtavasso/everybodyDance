"""Song-builder: dance continuously, the song builds itself.

No controls, no selection -- the instruments are authored in a fixed order, the
phases auto-advance to the clock, and each committed stem becomes a looping
skeleton "ghost" of the movement that made it. You just keep dancing and a
chorus of your past selves accumulates underneath you.

Per instrument, two deterministic passes (this is the legibility fix -- onsets
and pitch are body-authored, not Euclid/RNG/chord-nudge):

  RHYTHM pass  -- your HITS (sharp extremity strikes) place note onsets on the
                  grid. You choose where they land.
  PITCH pass   -- the rhythm loops back; your whole-body HEIGHT (crouch->rise)
                  sets each onset's pitch as it replays (snapped to scale for
                  safety, but your contour). Drums use height -> kick/snare/hat.

Timeline:  count-in -> [drums R,P] -> [bass R,P] -> [keys R,P] -> [lead R,P] -> loop
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .features import FeatureExtractor
from .laban import LabanEstimator
from .oscillator import EntrainedClock
from .output import Backend, MusicEvent
from .personalization import Normalizer, Profile
from .pose import JOINT_INDEX
from .substrate import Substrate, biased_config

CHAN = {"drums": 10, "bass": 1, "keys": 2, "lead": 3}
DRUM_PIECE = {0: 36, 1: 38, 2: 42}          # low/mid/high -> kick/snare/hat
DUR = {"bass": 0.9, "keys": 1.4, "lead": 0.35}
GHOST_SLOTS = 64                             # poses kept per looping ghost


@dataclass
class BuilderConfig:
    order: List[str] = field(default_factory=lambda: ["drums", "bass", "keys", "lead"])
    steps_per_beat: int = 4
    beats_per_bar: int = 4
    loop_bars: int = 2
    bars_rhythm: int = 2
    bars_pitch: int = 2
    count_in_bars: int = 1

    @property
    def steps_per_bar(self) -> int:
        return self.steps_per_beat * self.beats_per_bar

    @property
    def loop_steps(self) -> int:
        return self.steps_per_bar * self.loop_bars


@dataclass
class Slot:
    vel: int
    height01: float = 0.5
    pitch: Optional[int] = None


@dataclass
class UIState:
    order: List[str]
    status: Dict[str, str]          # role -> pending|rhythm|pitch|saved
    active: Optional[str]
    phase: str                      # count_in|rhythm|pitch|loop
    bars_left: int
    playhead: int
    onsets: Dict[str, List[bool]]
    pitch01: float
    live_xyz: np.ndarray            # current skeleton (normalised, y-up)
    ghosts: Dict[str, Optional[np.ndarray]]   # role -> pose at current loop phase
    bpm: float = 0.0


class SongBuilder:
    def __init__(self, backend: Backend, profile: Profile, tempo_hz: float,
                 hit_threshold: float, cfg: Optional[BuilderConfig] = None):
        self.backend = backend
        self.cfg = cfg or BuilderConfig()
        self.sub = Substrate(biased_config(profile.signature))
        self.sub.cfg.steps_per_beat = self.cfg.steps_per_beat
        self.sub.cfg.beats_per_bar = self.cfg.beats_per_bar
        self.features = FeatureExtractor()
        self.laban = LabanEstimator()
        self.norm = Normalizer(profile)
        self.clock = EntrainedClock(tempo_mode="fixed", fixed_hz=tempo_hz,
                                    steps_per_beat=self.cfg.steps_per_beat)
        self._hit_thr = hit_threshold

        L = self.cfg.loop_steps
        self.loops: Dict[str, List[Optional[Slot]]] = {r: [None] * L for r in self.cfg.order}
        self.status: Dict[str, str] = {r: "pending" for r in self.cfg.order}
        # looping skeleton ghosts, captured per loop-phase slot
        self.ghosts: Dict[str, List[Optional[np.ndarray]]] = {
            r: [None] * GHOST_SLOTS for r in self.cfg.order}

        self._idx = -1                 # current instrument index (-1 = count-in)
        self.phase = "count_in"
        self._bar_in_phase = 0
        self._abs_step = -1
        self._master = 0
        self._frac = 0.0
        self._hit_refractory = 0.0
        self._hit_this_step = (0.0, 0.5)
        self._offs: List = []
        self._seq = 0
        self._t = 0.0

    @property
    def active(self) -> Optional[str]:
        if self.phase in ("rhythm", "pitch") and 0 <= self._idx < len(self.cfg.order):
            return self.cfg.order[self._idx]
        return None

    @property
    def done(self) -> bool:
        return self.phase == "loop"

    # -- main step ---------------------------------------------------------

    def step(self, frame) -> UIState:
        self._t = frame.t
        feats = self.features.update(frame)
        drivers = self.laban.update(feats, feats.dt)
        clk = self.clock.update(float(feats.bounce), feats.dt)
        eff = self.norm.effort(drivers)
        nf = self.norm.features(feats, eff)
        height01 = float(np.clip(nf.com_height, 0, 1))

        self._frac = (clk.phase / (2 * np.pi) * self.cfg.steps_per_beat) % 1.0
        self._detect_hit(feats, height01)

        if clk.step_advanced:
            self._abs_step += 1
            self._master = self._abs_step % self.cfg.loop_steps
            self._on_step(feats, eff, nf, height01)

        # capture the looping skeleton ghost for the active instrument
        if self.active is not None:
            mfrac = (self._master + self._frac) / self.cfg.loop_steps
            self.ghosts[self.active][int(mfrac * GHOST_SLOTS) % GHOST_SLOTS] = \
                frame.xyz.copy()

        self._flush_offs(frame.t)
        return self._ui(frame.xyz, height01, clk)

    # -- hit detection -----------------------------------------------------

    def _detect_hit(self, feats, height01) -> None:
        idx = [JOINT_INDEX[j] for j in ("l_wrist", "r_wrist", "l_ankle", "r_ankle")]
        strength = float(feats.accel_mag[idx].max())
        if self._hit_refractory > 0:
            self._hit_refractory -= feats.dt
        if strength > self._hit_thr and self._hit_refractory <= 0:
            self._hit_refractory = 0.12
            if strength > self._hit_this_step[0]:
                self._hit_this_step = (strength, height01)

    # -- per-step FSM + authoring ------------------------------------------

    def _on_step(self, feats, eff, nf, height01) -> None:
        step = self._master
        n = len(self.sub.cfg.harmonic_field)
        self.sub.set_field_index((step * n) // self.cfg.loop_steps)

        if step % self.cfg.steps_per_bar == 0:
            self._advance_phase()

        for role in self.cfg.order:                 # play every saved stem
            if self.status[role] == "saved":
                self._play(role, step, feats.t)

        if self.phase == "rhythm":
            self._author_rhythm(step)
        elif self.phase == "pitch":
            self._author_pitch(step, height01, feats.t)

        self._hit_this_step = (0.0, 0.5)

    def _advance_phase(self) -> None:
        # only act on whole bars; also only on the loop downbeat for clean starts
        self._bar_in_phase += 1
        if self.phase == "count_in":
            if self._bar_in_phase >= self.cfg.count_in_bars and self._master == 0:
                self._start_instrument(0)
        elif self.phase == "rhythm":
            if self._bar_in_phase >= self.cfg.bars_rhythm:
                self.phase = "pitch"
                self.status[self.active] = "pitch"
                self._bar_in_phase = 0
        elif self.phase == "pitch":
            if self._bar_in_phase >= self.cfg.bars_pitch:
                self._save_and_advance()

    def _start_instrument(self, idx: int) -> None:
        self._idx = idx
        self.phase = "rhythm"
        self._bar_in_phase = 0
        role = self.cfg.order[idx]
        self.status[role] = "rhythm"
        self.loops[role] = [None] * self.cfg.loop_steps

    def _save_and_advance(self) -> None:
        role = self.cfg.order[self._idx]
        for slot in self.loops[role]:
            if slot is not None and slot.pitch is None and role != "drums":
                slot.pitch = self._pitch(role, slot.height01)
        self.status[role] = "saved"
        if self._idx + 1 < len(self.cfg.order):
            self._start_instrument(self._idx + 1)
        else:
            self.phase = "loop"           # everything authored: keep looping

    def _author_rhythm(self, step) -> None:
        strength, height01 = self._hit_this_step
        if strength > 0:
            vel = int(np.clip(55 + 12 * strength ** 0.5, 1, 127))
            self.loops[self.active][step] = Slot(vel=vel, height01=height01)
            self._play(self.active, step, self._t, live=height01, tag_live=True)

    def _author_pitch(self, step, height01, t) -> None:
        slot = self.loops[self.active][step]
        if slot is not None:
            slot.height01 = height01      # sculpt this onset's pitch by height
            self._play(self.active, step, t, live=height01, tag_live=True)

    # -- voicing -----------------------------------------------------------

    def _pitch(self, role, height01) -> int:
        r = self.sub.cfg.roles["lead" if role == "lead" else
                               "chord" if role == "keys" else "bass"]
        return self.sub.snap(float(np.clip(height01, 0, 1)), r, chord_weighted=False)

    def _play(self, role, step, t, live=None, tag_live=False) -> None:
        slot = self.loops[role][step]
        if slot is None:
            return
        if role == "drums":
            note, dur = DRUM_PIECE[int(np.clip(slot.height01 * 3, 0, 2))], 0.07
        else:
            h = live if live is not None else slot.height01
            note = slot.pitch if slot.pitch is not None else self._pitch(role, h)
            dur = DUR[role]
        self._emit(MusicEvent("note_on", CHAN[role], int(note), slot.vel,
                              dur=dur, tag=role + (".live" if tag_live else "")), t)

    def _emit(self, ev, t) -> None:
        ev.t = t
        self.backend.send(ev)
        if ev.kind == "note_on" and ev.dur:
            self._seq += 1
            heapq.heappush(self._offs, (t + ev.dur, self._seq,
                                        MusicEvent("note_off", ev.channel, ev.a, 0,
                                                   t=t + ev.dur, tag=ev.tag)))

    def _flush_offs(self, t) -> None:
        while self._offs and self._offs[0][0] <= t:
            self.backend.send(heapq.heappop(self._offs)[2])

    # -- UI snapshot -------------------------------------------------------

    def _ui(self, live_xyz, height01, clk) -> UIState:
        bars_total = {"rhythm": self.cfg.bars_rhythm, "pitch": self.cfg.bars_pitch,
                      "count_in": self.cfg.count_in_bars}.get(self.phase, 0)
        mfrac = (self._master + self._frac) / self.cfg.loop_steps
        gslot = int(mfrac * GHOST_SLOTS) % GHOST_SLOTS
        ghosts = {r: (self.ghosts[r][gslot] if self.status[r] == "saved" else None)
                  for r in self.cfg.order}
        onsets = {r: [s is not None for s in self.loops[r]] for r in self.cfg.order}
        return UIState(
            order=list(self.cfg.order), status=dict(self.status), active=self.active,
            phase=self.phase, bars_left=max(0, bars_total - self._bar_in_phase),
            playhead=self._master, onsets=onsets, pitch01=height01,
            live_xyz=live_xyz, ghosts=ghosts, bpm=clk.bpm)
