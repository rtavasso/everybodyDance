"""The Studio -- a deterministic, multi-stem, gesture-driven performance engine.

One process per :meth:`Studio.step`: pose -> features + Effort -> entrained clock
-> a flat signal dict (the keys :mod:`mapping` consumes) -> per-stem continuous
targets + discrete gesture actions -> in-scale, step-quantised MIDI on five stems
(drums/bass/keys/lead/texture). It is the band: every stem plays at once, the
body shapes them all continuously, named moves punch in salient changes, and any
stem can be looped with the body-looper.

Two things make it feel *caused and alive* rather than a backing track:

  * the **continuous** mapping (every signal keeps shaping every stem), and
  * the **phantom gate**: new note onsets are suppressed when the body is still
    or absent, so a freeze actually goes quiet (notes ring out via the note_off
    scheduler) instead of the grid manufacturing phantom hits.

Determinism is sacred: the same dance + same commands replays byte-identically.
Any RNG is seeded from ``cfg.seed`` and only used for taste, never structure.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from . import stems as stemlib
from .effects import STYLE, EffectEngine, Flash
from .features import FeatureExtractor
from .gestures import GestureRecognizer, build_gestures
from .laban import LabanEstimator
from .mapping import MappingConfig, MappingResolver
from .oscillator import EntrainedClock
from .output import Backend, MusicEvent
from .personalization import Normalizer, Profile
from .pose import JOINTS
from .substrate import SCALES, Substrate, biased_config


@dataclass
class StudioConfig:
    steps_per_beat: int = 4
    beats_per_bar: int = 4
    loop_bars: int = 2
    coupling: float = 0.4
    seed: int = 0
    motion_floor: float = 0.06   # normalized energy below this => stillness gate

    @property
    def steps_per_bar(self) -> int:
        return self.steps_per_beat * self.beats_per_bar

    @property
    def loop_steps(self) -> int:
        return self.steps_per_bar * self.loop_bars


@dataclass
class StemUI:
    name: str
    color: tuple
    level: float
    active: bool
    looping: bool
    recording: bool
    muted: bool
    density: float
    cutoff: float
    onsets: list
    timbre: str


@dataclass
class StudioUI:
    t: float
    phase: str
    bpm: float
    present: bool
    energy: float
    live_xyz: object
    stems: list
    playhead: int
    loop_steps: int
    flashes: list
    prompt: object
    mood: str
    beat: bool


@dataclass
class StudioOut:
    ui: StudioUI
    automation: dict


# A small cycling onboarding hint -> shown until the body is dancing.
PROMPTS = ["JUMP -> fill", "T-POSE -> breakdown", "HANDS UP -> build",
           "SQUAT -> loop bass", "ARMS CROSSED -> dark mood", "CLAP -> snare"]


class Studio:
    def __init__(self, backend: Backend, profile: Profile,
                 mapping: Optional[MappingConfig] = None,
                 cfg: Optional[StudioConfig] = None):
        self.backend = backend
        self.cfg = cfg or StudioConfig()
        self.mapping = mapping or MappingConfig.default()
        self.resolver = MappingResolver(self.mapping)

        # Substrate (per-person biased) -- exposed as self.sub like engine.substrate.
        self.sub = Substrate(biased_config(profile.signature))
        self.sub.cfg.steps_per_beat = self.cfg.steps_per_beat
        self.sub.cfg.beats_per_bar = self.cfg.beats_per_bar

        # Upstream signal stack.
        self.features = FeatureExtractor()
        self.laban = LabanEstimator()
        self.clock = EntrainedClock(coupling=self.cfg.coupling,
                                    steps_per_beat=self.cfg.steps_per_beat)
        self.norm = Normalizer(profile)
        self.rng = np.random.default_rng(self.cfg.seed)

        # Stems + the gesture/effects layers.
        timbres = {m.stem: m.timbre for m in self.mapping.stems}
        self.rack = stemlib.StemRack(self.sub, self.cfg.loop_steps, timbres)
        self.recognizer = GestureRecognizer(build_gestures())
        self.fx = EffectEngine(self.backend, self.sub)

        # Loop grid + scheduler state.
        self._abs_step = -1
        self.master_step = 0
        self._offs: List = []
        self._seq = 0
        self._t = 0.0
        self.bpm = 0.0
        self.mood = self.sub.cfg.scale
        self._prompt_idx = 0

        # transient gesture windows (in abs_step units), like looper.py
        self._fill_until = -1
        self._build_until = -1
        self._drop_until = -1
        self._breakdown_until = -1
        self._still = False           # are we in a stillness window?

    # -- main step ---------------------------------------------------------

    def step(self, frame, commands: Optional[List[str]] = None) -> StudioOut:
        commands = commands or []
        self._t = frame.t
        feats = self.features.update(frame)
        drivers = self.laban.update(feats, feats.dt)
        clk = self.clock.update(float(feats.bounce), feats.dt)
        eff = self.norm.effort(drivers)
        nf = self.norm.features(feats, eff)
        sig = self._signal(feats, eff, nf)
        present = bool(feats.present)
        self.bpm = clk.bpm

        # 1) continuous per-stem targets
        params = self.resolver.resolve_continuous(sig)
        for stem in self.rack:
            stem.params = params[stem.name]
            if stem.timbre == "":
                stem.timbre = stem.params.timbre

        # 2) gestures: recognized + force-fired commands -> actions + flashes
        moves = [g.name for g in self.recognizer.update(frame, feats)]
        moves += list(commands)
        for move in moves:
            self._apply_move(move, frame.t)

        # 3) stillness / presence gate (the phantom fix)
        gated = (sig["energy"] < self.cfg.motion_floor) or not present

        # 4) loop grid advance
        beat = bool(clk.beat)
        if clk.step_advanced:
            self._abs_step += 1
            self.master_step = self._abs_step % self.cfg.loop_steps
            self._on_step(sig, eff, nf, clk, gated)

        for stem in self.rack:
            stem.decay_level(feats.dt)
        self._flush_offs(frame.t)

        ui = self._ui(frame, sig, present, gated, clk, beat)
        return StudioOut(ui=ui, automation=self._automation())

    # -- signal dict (the keys mapping.py expects) -------------------------

    def _signal(self, feats, eff, nf) -> dict:
        ev = feats.events
        return {
            "energy": float(nf.energy),
            "core_energy": float(nf.core_energy),
            "limb_energy": float(nf.limb_energy),
            "openness": float(nf.openness),
            "com_height": float(nf.com_height),
            "weight": float(eff.weight),
            "time": float(eff.time),
            "space": float(eff.space),
            "flow": float(eff.flow),
            "asymmetry_lr": float(feats.asymmetry_lr),
            "thrust": float(ev.get("thrust", 0.0)),
            "stomp": float(ev.get("stomp", 0.0)),
            "reversal": float(ev.get("reversal", 0.0)),
            "freeze": float(ev.get("freeze", 0.0)),
        }

    # -- per-step generation -----------------------------------------------

    def _on_step(self, sig, eff, nf, clk, gated) -> None:
        step = self.master_step
        # deterministic harmony as a function of loop position (looper/builder).
        n = len(self.sub.cfg.harmonic_field)
        self.sub.set_field_index((step * n) // self.cfg.loop_steps)
        beat_s = 60.0 / max(clk.bpm, 1.0)
        steps_per_bar = self.cfg.steps_per_bar

        entering_still = gated and not self._still
        self._still = gated
        filling = (self._abs_step < self._fill_until)

        for stem in self.rack:
            stem.active = False
            audible = self.rack.audible(stem)
            events: List[MusicEvent] = []

            # Locked loops replay deterministically (still subject to mute/solo).
            if stem.looping:
                events = stem.replay(step)
            else:
                density = self._density(stem, sig, step)
                if gated:
                    # PHANTOM GATE: no new onsets while still/absent. Allow ONE
                    # soft sustained pad revoice at the moment stillness begins,
                    # so it's never abruptly cut -- then nothing more.
                    if entering_still and stem.name in ("keys", "texture"):
                        events = stemlib.revoice_pad(stem, self.sub, sig, beat_s)
                elif density > 0.0:
                    events = stemlib.generate(stem, self.sub, step, steps_per_bar,
                                              sig, beat_s, density)
                # FILL: a guaranteed short percussion burst over the fill window.
                if filling and stem.name == "drums" and not gated:
                    events = events + [MusicEvent("note_on", stem.channel,
                                                  stemlib.DRUM_PIECE["hat"], 78,
                                                  dur=stemlib.DUR["drums"], tag="drums")]
                stem.capture(step, events)

            if events:
                stem.active = True
                stem.level = max(stem.level, max(
                    (e.b for e in events if e.kind == "note_on"), default=0) / 127.0)
            if audible:
                for ev in events:
                    self._emit(ev, self._t)

    def _density(self, stem, sig, step) -> float:
        """Mapped density, modulated by the active fill/build/drop windows."""
        d = float(stem.params.density)
        a = self._abs_step + 1                     # density is decided for next step
        if stem.name in ("drums",) and a < self._fill_until:
            d = min(1.0, d + 0.4)                  # fill: extra percussion
        if a < self._build_until:
            # build: rising density ramp toward the end of the window
            frac = 1.0 - (self._build_until - a) / max(self.cfg.steps_per_bar, 1)
            d = min(1.0, d + 0.5 * float(np.clip(frac, 0, 1)))
        if a < self._breakdown_until and stem.name in ("drums", "bass"):
            d = 0.0                                # breakdown: strip the bottom
        if a < self._drop_until and stem.name in ("lead", "keys", "texture"):
            d = 0.0                                # drop: mute the top, then back
        return d

    # -- gesture actions ---------------------------------------------------

    def _apply_move(self, move: str, t: float) -> None:
        # Always flash so the stage can show any recognized/forced move.
        color, big = STYLE.get(move, ((230, 230, 230), False))
        self.fx.flashes.append(Flash(move, color, t, 0.5 if big else 0.4, big))

        for g in self.resolver.resolve_gesture(move):
            self._do_action(g, t)

    def _do_action(self, g, t: float) -> None:
        action, target, params = g.action, g.target, g.params
        if action == "fx":
            # One-shot hit on the target stem; EffectEngine tags "fx" + carries
            # its own channel. Map the binding's params to a known fx move.
            self.fx.trigger(_fx_move(params), t)
        elif action == "fill":
            self._fill_until = self._abs_step + self.cfg.steps_per_bar
        elif action == "drop":
            self._drop_until = self._abs_step + self.cfg.steps_per_bar
        elif action == "breakdown":
            self._breakdown_until = self._abs_step + self.cfg.steps_per_bar
        elif action == "build":
            self._build_until = self._abs_step + self.cfg.steps_per_bar
        elif action == "loop_record":
            s = self.rack.get(target)
            if s is not None:
                s.start_record()
        elif action == "loop_toggle":
            s = self.rack.get(target)
            if s is not None:
                s.toggle_loop()
        elif action == "loop_clear":
            s = self.rack.get(target)
            if s is not None:
                s.clear_loop()
        elif action == "scale_shift":
            scale = self.mapping.moods.get(target)
            if scale in SCALES:
                self.sub.cfg.scale = scale
                self.sub._scale = SCALES[scale]
                self.mood = scale
        elif action == "timbre_morph":
            s = self.rack.get(target)
            to = params.get("to")
            if s is not None and to:
                s.timbre = to
        elif action == "stem_mute":
            s = self.rack.get(target)
            if s is not None:
                s.muted = not s.muted
        elif action == "stem_solo":
            s = self.rack.get(target)
            if s is not None:
                s.soloed = not s.soloed

    # -- scheduler (heapq note_off, exactly like engine/builder/looper) ----

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

    def panic(self) -> None:
        """Flush every scheduled note_off, then all-notes-off. No hanging notes."""
        while self._offs:
            self.backend.send(heapq.heappop(self._offs)[2])
        self.backend.panic()

    # -- UI + automation ---------------------------------------------------

    def _automation(self) -> dict:
        out: dict = {}
        for stem in self.rack:
            p = stem.params
            out[stem.name] = {
                "cutoff": float(p.cutoff), "drive": float(p.drive),
                "gain": float(p.gain), "pan": float(p.pan),
                "reverb": float(p.reverb), "delay": float(p.delay),
                "timbre": stem.timbre or p.timbre,
            }
        return out

    def _ui(self, frame, sig, present, gated, clk, beat) -> StudioUI:
        live = (frame.xyz if present and frame.xyz is not None
                else np.zeros((len(JOINTS), 3)))
        stem_ui = []
        for stem in self.rack:
            stem_ui.append(StemUI(
                name=stem.name, color=stem.color, level=float(np.clip(stem.level, 0, 1)),
                active=bool(stem.active), looping=bool(stem.looping),
                recording=bool(stem.recording),
                muted=not self.rack.audible(stem),
                density=float(stem.params.density), cutoff=float(stem.params.cutoff),
                onsets=stem.loop_onsets(), timbre=stem.timbre or stem.params.timbre))
        phase = "perform" if present else "attract"
        prompt = None
        if not present or sig["energy"] < self.cfg.motion_floor:
            prompt = PROMPTS[self._prompt_idx % len(PROMPTS)]
            self._prompt_idx += 1
        return StudioUI(
            t=frame.t, phase=phase, bpm=float(self.bpm), present=present,
            energy=float(sig["energy"]), live_xyz=live, stems=stem_ui,
            playhead=int(self.master_step), loop_steps=int(self.cfg.loop_steps),
            flashes=list(self.fx.active_flashes(frame.t)), prompt=prompt,
            mood=self.mood, beat=bool(beat))


def _fx_move(params: dict) -> str:
    """Map an fx binding's params to a move name EffectEngine knows."""
    hit = (params or {}).get("hit", "")
    return {"snare": "CLAP", "stab": "PUNCH"}.get(hit, "CLAP")
