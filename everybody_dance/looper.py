"""Live body-looper -- build each instrument's loop with your body, commit it
with a pedal, layer the next one, then once everything is locked, let the dance
modulate the whole gestalt. (Imogen Heap's gloves, but whole-body.)

Flow:
    RECORD drums -> [pedal] -> RECORD bass -> [pedal] -> RECORD chord
        -> [pedal] -> RECORD lead -> [pedal] -> PERFORM

  * tempo entrains to the body while recording the first instrument, then LOCKS
    on the first commit so every later loop shares one grid;
  * committed loops play back deterministically while you record the next track;
  * in PERFORM the body shapes the gestalt -- continuously (energy/posture ->
    master filter, intensity, density) and via gestures (thrust=fill, stomp=drop,
    freeze=breakdown, opening up=build).

Loops are step-quantised to the entrained grid, so they stay locked and musical.
"""

from __future__ import annotations

import copy
import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .controls import Command, ControlInput, ScriptedControl
from .features import FeatureExtractor
from .laban import LabanEstimator
from .oscillator import EntrainedClock
from .output import Backend, MusicEvent
from .personalization import Normalizer, Profile
from .substrate import Substrate, biased_config

MASTER_CH = 16
CC_CUTOFF, CC_EXPRESSION, CC_MOD = 74, 11, 1


@dataclass
class LooperConfig:
    steps_per_beat: int = 4
    beats_per_bar: int = 4
    loop_bars: int = 2
    record_order: List[str] = field(
        default_factory=lambda: ["drums", "bass", "chord", "lead"])
    coupling: float = 0.4
    seed: int = 0

    @property
    def steps_per_bar(self) -> int:
        return self.steps_per_beat * self.beats_per_bar

    @property
    def loop_steps(self) -> int:
        return self.steps_per_bar * self.loop_bars


@dataclass
class LooperState:
    phase: str = "record"          # "record" | "perform"
    active: Optional[str] = None   # role currently being recorded
    committed: List[str] = field(default_factory=list)
    tempo_locked: bool = False
    bpm: float = 0.0
    master_step: int = 0


class LoopStation:
    def __init__(self, backend: Backend, profile: Profile,
                 cfg: Optional[LooperConfig] = None):
        self.backend = backend
        self.cfg = cfg or LooperConfig()
        self.sub = Substrate(biased_config(profile.signature))
        self.sub.cfg.steps_per_beat = self.cfg.steps_per_beat
        self.sub.cfg.beats_per_bar = self.cfg.beats_per_bar
        self.sub.cfg.roles["bass"].max_density = max(
            6, self.sub.cfg.roles["bass"].max_density)
        self.features = FeatureExtractor()
        self.laban = LabanEstimator()
        self.clock = EntrainedClock(coupling=self.cfg.coupling,
                                    steps_per_beat=self.cfg.steps_per_beat)
        self.norm = Normalizer(profile)
        self.rng = np.random.default_rng(self.cfg.seed)

        self.st = LooperState(active=self.cfg.record_order[0])
        L = self.cfg.loop_steps
        self.buffers: Dict[str, List[List[MusicEvent]]] = {
            r: [[] for _ in range(L)] for r in self.cfg.record_order}

        self._abs_step = -1
        self._offs: List = []
        self._seq = 0
        self._cc_last: Dict[int, int] = {}
        # perform-mode transient gesture state
        self._fill_until = -1
        self._drop_until = -1
        self._build = 0.0

    # -- public ------------------------------------------------------------

    def run(self, source, control: Optional[ControlInput] = None):
        control = control or ScriptedControl([])
        for frame in source.frames():
            cmds = control.poll(frame.t)
            self.step(frame, cmds)
        self.backend.panic()

    def step(self, frame, commands: List[Command]) -> None:
        feats = self.features.update(frame)
        drivers = self.laban.update(feats, feats.dt)
        clk = self.clock.update(float(feats.bounce), feats.dt)
        eff = self.norm.effort(drivers)
        nf = self.norm.features(feats, eff)

        for c in commands:
            self._handle(c, clk)

        self.st.bpm = clk.bpm
        # advance the loop grid
        if clk.step_advanced:
            self._abs_step += 1
            self.st.master_step = self._abs_step % self.cfg.loop_steps
            self._on_step(feats, eff, nf, clk)

        # continuous, per-frame modulation lanes
        self._continuous(feats, eff, nf)
        self._flush_offs(feats.t)

    # -- control -----------------------------------------------------------

    def _handle(self, c: Command, clk) -> None:
        if c == Command.COMMIT:
            self._commit(clk)
        elif c == Command.UNDO:
            self._undo()
        elif c == Command.RESET:
            self._reset()

    def _commit(self, clk) -> None:
        if self.st.phase != "record" or self.st.active is None:
            return
        role = self.st.active
        if role not in self.st.committed:
            self.st.committed.append(role)
        # Lock tempo on the first commit so every later loop shares the grid.
        if not self.st.tempo_locked:
            self.clock.tempo_mode = "fixed"
            self.clock.fixed_w = max(clk.tempo_hz, 0.5) * 2 * np.pi
            self.st.tempo_locked = True
        # advance
        idx = self.cfg.record_order.index(role) + 1
        if idx >= len(self.cfg.record_order):
            self.st.active = None
            self.st.phase = "perform"
        else:
            self.st.active = self.cfg.record_order[idx]

    def _undo(self) -> None:
        # clear the current take (re-record the active track)
        target = self.st.active or (self.st.committed[-1] if self.st.committed else None)
        if target is None:
            return
        self.buffers[target] = [[] for _ in range(self.cfg.loop_steps)]
        if self.st.phase == "perform":
            self.st.phase = "record"
        self.st.active = target
        if target in self.st.committed:
            self.st.committed.remove(target)

    def _reset(self) -> None:
        L = self.cfg.loop_steps
        self.buffers = {r: [[] for _ in range(L)] for r in self.cfg.record_order}
        self.st = LooperState(active=self.cfg.record_order[0])
        self.clock.tempo_mode = "entrain"
        self._abs_step = -1

    # -- per-step emission -------------------------------------------------

    def _on_step(self, feats, eff, nf, clk) -> None:
        step = self.st.master_step
        # deterministic harmony as a function of loop position
        n = len(self.sub.cfg.harmonic_field)
        self.sub.set_field_index((step * n) // self.cfg.loop_steps)

        if self.st.phase == "record":
            # committed tracks play back; active track is live + recorded
            for role in self.st.committed:
                for ev in self.buffers[role][step]:
                    self._emit(copy.copy(ev), feats.t)
            if self.st.active is not None:
                evs = self._generate(self.st.active, step, eff, nf, clk)
                self.buffers[self.st.active][step] = evs
                for ev in evs:
                    self._emit(copy.copy(ev), feats.t)
        else:
            self._perform_step(step, feats, eff, nf, clk)

    def _perform_step(self, step, feats, eff, nf, clk) -> None:
        intensity = 0.55 + 0.45 * nf.energy           # gestalt breathes with energy
        dropped = self._abs_step < self._drop_until
        breakdown = feats.events.get("freeze", 0.0) > 0.6
        for role in self.st.committed:
            if dropped and role in ("lead", "chord"):
                continue
            if breakdown and role in ("drums", "bass"):
                continue
            for ev in self.buffers[role][step]:
                e = copy.copy(ev)
                if e.kind == "note_on":
                    # density gate: thin hats when energy is low
                    if e.tag == "drums.hat" and self.rng.random() > 0.3 + 0.7 * nf.energy:
                        continue
                    e.b = int(np.clip(e.b * intensity, 1, 127))
                self._emit(e, feats.t)
        self._perform_gestures(step, feats, eff, nf, clk)

    def _perform_gestures(self, step, feats, eff, nf, clk) -> None:
        ev = feats.events
        if ev.get("thrust", 0.0) > 0.5 or ev.get("reversal", 0.0) > 0.6:
            self._fill_until = self._abs_step + self.cfg.steps_per_bar
        if ev.get("stomp", 0.0) > 0.5:
            self._drop_until = self._abs_step + self.cfg.steps_per_bar
        # FILL: extra hats + a lead echo while active
        if self._abs_step < self._fill_until:
            self._emit(MusicEvent("note_on", 10, 42, 70, dur=0.05, tag="drums.hat"),
                       feats.t)
            if step % 2 == 0:
                note = self.sub.snap(0.6 + 0.4 * nf.openness,
                                     self.sub.cfg.roles["lead"])
                self._emit(MusicEvent("note_on", self.sub.cfg.roles["lead"].channel,
                                      note, 90, dur=0.12, tag="lead.fill"), feats.t)

    # -- per-role generation (body -> that instrument) ---------------------

    def _generate(self, role, step, eff, nf, clk) -> List[MusicEvent]:
        bar_step = step % self.cfg.steps_per_bar
        beat_s = 60.0 / max(clk.bpm, 1.0)
        # While RECORDING a track you're deliberately performing THAT instrument,
        # so density has a floor and follows overall energy -- every take lays
        # down notes; the body modulates dynamics/articulation/pitch around it.
        e = nf.energy
        out: List[MusicEvent] = []
        if role == "drums":
            kick = self.sub.pattern("drums", 0.4 + 0.5 * e)
            hat = self.sub.pattern("drums", 0.5 + 0.5 * e)
            if kick[bar_step] and self.rng.random() < 0.7 + 0.3 * eff.weight:
                out.append(MusicEvent("note_on", 10, 36,
                                      int(np.clip(60 + 60 * eff.weight, 1, 127)),
                                      dur=0.08, tag="drums.kick"))
            if hat[bar_step] and self.rng.random() < 0.5 + 0.4 * e:
                out.append(MusicEvent("note_on", 10, 42,
                                      int(np.clip(40 + 50 * eff.time, 1, 127)),
                                      dur=0.05, tag="drums.hat"))
        elif role == "bass":
            bass = self.sub.pattern("bass", 0.5 + 0.4 * e)
            if bass[bar_step]:
                r = self.sub.cfg.roles["bass"]
                note = self.sub.fold_into_range(
                    self.sub.degree_to_midi(self.sub.chord_root_degree, octave=-1),
                    r.lo, r.hi)
                dur = beat_s * (0.9 if eff.flow > 0.5 else 0.4)
                out.append(MusicEvent("note_on", r.channel, note,
                                      int(np.clip(55 + 55 * eff.weight, 1, 127)),
                                      dur=dur, tag="bass"))
        elif role == "chord":
            if bar_step == 0:                       # revoice once per bar
                r = self.sub.cfg.roles["chord"]
                spread = 0.3 + 0.7 * nf.openness
                base_oct = int(round(nf.com_height))
                notes = sorted({self.sub.fold_into_range(
                    self.sub.degree_to_midi(d, octave=base_oct + int(k * spread)),
                    r.lo, r.hi) for k, d in enumerate(self.sub.chord_tones())})
                vel = int(np.clip(35 + 45 * eff.weight, 1, 100))
                bar_s = beat_s * self.cfg.beats_per_bar
                for nnote in notes:
                    out.append(MusicEvent("note_on", r.channel, nnote, vel,
                                          dur=bar_s * 0.98, tag="chord"))
        elif role == "lead":
            p = 0.3 + 0.6 * e * (0.5 + 0.5 * eff.space)
            if self.rng.random() < p:
                r = self.sub.cfg.roles["lead"]
                field = float(np.clip(0.5 * nf.openness + 0.5 * nf.com_height, 0, 1))
                note = self.sub.snap(field, r, chord_weighted=eff.space > 0.4)
                dur = beat_s * (0.15 + 0.7 * eff.flow)
                out.append(MusicEvent("note_on", r.channel, note,
                                      int(np.clip(50 + 50 * eff.weight, 1, 127)),
                                      dur=dur, tag="lead"))
        return out

    # -- continuous modulation + scheduling --------------------------------

    def _continuous(self, feats, eff, nf) -> None:
        if self.st.phase != "perform":
            return
        # whole-body energy/posture -> master filter, expression, colour
        self._cc(CC_CUTOFF, int(30 + 95 * (0.5 * nf.energy + 0.5 * eff.weight)), feats.t)
        self._cc(CC_EXPRESSION, int(40 + 87 * nf.energy), feats.t)
        self._cc(CC_MOD, int(127 * nf.openness), feats.t)

    def _cc(self, num: int, value: float, t: float) -> None:
        v = int(np.clip(value, 0, 127))
        if abs(self._cc_last.get(num, -999) - v) < 2:
            return
        self._cc_last[num] = v
        self.backend.send(MusicEvent("cc", MASTER_CH, num, v, t=t, tag=f"cc.{num}"))

    def _emit(self, ev: MusicEvent, t: float) -> None:
        ev.t = t
        self.backend.send(ev)
        if ev.kind == "note_on" and ev.dur:
            self._seq += 1
            heapq.heappush(self._offs, (t + ev.dur, self._seq,
                                        MusicEvent("note_off", ev.channel, ev.a, 0,
                                                   t=t + ev.dur, tag=ev.tag)))

    def _flush_offs(self, t: float) -> None:
        while self._offs and self._offs[0][0] <= t:
            self.backend.send(heapq.heappop(self._offs)[2])
