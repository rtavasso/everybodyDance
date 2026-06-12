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

from . import groove as G
from . import stems as stemlib
from .effects import STYLE, EffectEngine, Flash
from .features import FeatureExtractor
from .game import (GOLD_COLOR, SECTIONS, ComboTracker, GameUI, GoldMoveGame,
                   SongArc, StreakMeter)
from .gestures import GestureRecognizer, build_gestures
from .identity import derive_identity
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
    still_dwell_s: float = 0.4   # low energy must HOLD this long to count as a
                                 # freeze (real dancers dip through the floor
                                 # constantly; a slow transition isn't a freeze)
    game: bool = True            # combos / streak / song arc / gold moves
    use_identity: bool = True    # Dance DNA: per-person key/scale/progression/kit

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
    locked: bool = False         # song-arc unlock state (game layer)


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
    game: object = None          # GameUI snapshot, or None when game is off
    still: bool = False          # the stillness gate's actual state this frame


@dataclass
class StudioOut:
    ui: StudioUI
    automation: dict


# A small cycling onboarding hint -> shown until the body is dancing.
PROMPTS = ["JUMP -> fill", "T-POSE -> breakdown", "HANDS UP -> build",
           "SQUAT -> loop bass", "ARMS CROSSED -> dark mood", "CLAP -> snare",
           "SQUAT then JUMP -> SUPERNOVA", "CLAP x3 -> CLAP STORM",
           "STOMP x2 -> EARTHQUAKE"]


class Studio:
    # SALIENCE CAP: a move a dancer makes constantly isn't a salient move any
    # more -- it's how they dance, and the continuous layer already expresses
    # it. Two rolling limits per move: a burst allowance (4 per 14 s -- keeps
    # the x3 combos fully responsive) and a sustained rate (6 per 40 s, i.e.
    # ~9/min -- holds real-dancer spam under the G4 <=20/min SLO).
    MOVE_CAP = 4
    MOVE_CAP_WINDOW = 14.0
    MOVE_CAP_SUSTAIN = 6
    MOVE_CAP_SUSTAIN_WINDOW = 40.0

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

        # Dance DNA: this person's key / mode / progression / kit / stage name.
        # Rides on top of biased_config (which keeps roles/densities) and
        # overrides the harmonic choices, so every visitor gets their own world.
        self.identity = (derive_identity(profile.signature)
                         if self.cfg.use_identity else None)
        if self.identity is not None:
            self.sub.cfg.tonic = self.identity.tonic
            self.sub.cfg.scale = self.identity.scale
            self.sub._scale = SCALES[self.identity.scale]
            self.sub.cfg.harmonic_field = list(self.identity.progression)
        # exact (t, tonic, scale) timeline -- the in-scale metric's authority.
        self.key_log: List = [(0.0, self.sub.cfg.tonic, self.sub.cfg.scale)]

        # Upstream signal stack.
        self.features = FeatureExtractor()
        self.laban = LabanEstimator()
        self.clock = EntrainedClock(coupling=self.cfg.coupling,
                                    steps_per_beat=self.cfg.steps_per_beat)
        self.norm = Normalizer(profile)
        self.rng = np.random.default_rng(self.cfg.seed)

        # Stems + the gesture/effects layers. The identity's kit takes
        # precedence over the mapping's default timbres (timbre_morph still
        # overrides either later).
        timbres = {m.stem: m.timbre for m in self.mapping.stems}
        if self.identity is not None:
            timbres.update(self.identity.kit)
        self.rack = stemlib.StemRack(self.sub, self.cfg.loop_steps, timbres)
        self.recognizer = GestureRecognizer(build_gestures())
        self.fx = EffectEngine(self.backend, self.sub)

        # The game layer: combos, streak, the song arc, gold-move challenges.
        self.combos = ComboTracker()
        self.streak = StreakMeter()
        self.arc = SongArc()
        self.gold = GoldMoveGame()
        self.combo_log: List = []                # [(t, combo name)]
        self.max_streak_tier = 0
        self._move_times: dict = {}              # move -> accepted fire times

        # The groove backbone: latched tempo, committed bar plans, the motif,
        # and the rhythmic-coherence skill loop (see groove.py).
        self.latch = G.TempoLatch(initial_hz=profile.char_tempo_hz)
        self.coherence = G.RhythmCoherence()
        self.flux = G.KineticFlux()              # body accents (onset detector)
        self.servo = G.PhaseServo()              # puts the beat ON the accents
        self.onset_log: List = []                # (t, strength) for metrics
        self._lean = 0.0                         # accent -> next beat leans in
        self._accent_last_t = -1e9
        self._last_pulse_t = None
        self.style = self.identity.style if self.identity else "backbeat"
        self.swing = G.SWING.get(self.style, 0.08)
        self.writer = G.MotifWriter(
            seed=self.cfg.seed + (self.identity.tonic if self.identity else 0))
        self.fx.quantize = self._quantize_fx     # fx one-shots land on a 16th
        self.grid_log: List = []                 # (abs_step, emit_t) for metrics
        self._next_step_t: Optional[float] = None
        self._contour: List[float] = []          # recent height contour (~1 bar)
        self._vel_bias = 0.0
        self._dens_acc: dict = {}                # per-stem bar-density average
        self._levels: dict = {}
        self._prev_levels: dict = {}
        self._drum_bar = None
        self._bass_bar = None
        self._keys_bar: dict = {}
        self._tex_bar: set = set()
        self._lead_on = False

        # Loop grid + scheduler state.
        self._abs_step = -1
        self.master_step = 0
        self._offs: List = []
        self._seq = 0
        self._t = 0.0
        self.bpm = 0.0
        self.mood = self.sub.cfg.scale
        self._base_scale = self.sub.cfg.scale     # ECLIPSE flips back to this
        self._prompt_idx = 0

        # transient gesture windows (in abs_step units), like looper.py
        # (_fill_until very negative so the first fill clears its budget)
        self._fill_until = -(10 ** 9)
        self._build_until = -1
        self._drop_until = -1
        self._breakdown_until = -1
        self._boost_until = -1        # combo payoff: everything denser/louder
        self._quake_until = -1        # combo payoff: bass an octave down, driven
        self._lifted = False          # PEAK has transposed the world +2
        self._still = False           # are we in a stillness window?
        self._still_since = None      # when energy first dipped under the floor

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

        # tempo: estimates only -- the latch decides, at bar lines. The pulse
        # listener's inter-peak estimate leads (robust on real bodies); the
        # entrained oscillator is the fallback before enough peaks exist.
        anchor = self._next_beat_t()
        # NEGATED bounce: the pulse listener detects the TROUGH -- the dig,
        # the bottom of the bounce, which is where a dancer marks the beat
        # (and which is sharp where the top is a plateau).
        self.coherence.update(-float(feats.bounce), frame.t, self.latch.beat_s,
                              feats.dt, anchor_t=anchor or 0.0)
        if present:
            est_hz = self.coherence.tempo_hz
            if est_hz is None and clk.confidence > 0.25:
                est_hz = clk.tempo_hz
            if est_hz is not None:
                self.latch.observe(est_hz, frame.t)
        self.bpm = self.latch.bpm
        # body accents: every significant movement is an onset the band hears
        # -- it feeds the phase servo (the downbeat moves TO your hits) and is
        # answered below once the stillness gate is known. The bounce pulse
        # itself also feeds the servo: for a smooth mover with no sharp
        # accents, the pulse IS the accent.
        onset = self.flux.update(feats.accel_mag, frame.t, feats.dt) \
            if present else None
        if onset is not None:
            self.onset_log.append((frame.t, onset))
            if anchor is not None:
                self.servo.observe(frame.t, onset, self.latch.beat_s, anchor)
            self._lean = max(self._lean, 16.0 * onset)
        pulse_t = self.coherence.last_peak_t
        if present and pulse_t is not None and pulse_t != self._last_pulse_t:
            self._last_pulse_t = pulse_t
            self.onset_log.append((pulse_t, 0.35))
            if anchor is not None:
                self.servo.observe(pulse_t, 0.35, self.latch.beat_s, anchor)
        self._contour.append(float(nf.com_height))
        if len(self._contour) > 64:
            self._contour.pop(0)

        # 1) continuous per-stem targets
        params = self.resolver.resolve_continuous(sig)
        if self._abs_step < self._quake_until:        # EARTHQUAKE payoff window
            qp = params.get("bass")
            if qp is not None:
                qp.register_octave -= 1
                qp.drive = float(min(1.0, qp.drive + 0.35))
        for stem in self.rack:
            stem.params = params[stem.name]
            if stem.timbre == "":
                stem.timbre = stem.params.timbre
            acc, cnt = self._dens_acc.get(stem.name, (0.0, 0))
            self._dens_acc[stem.name] = (acc + float(stem.params.density),
                                         cnt + 1)

        # game: heat / streak integrate every frame (moves arrive below).
        # Dancing ON THE BEAT is what heats the arc -- coherence scales energy,
        # so the path to PEAK runs through the groove (the skill loop).
        if self.cfg.game:
            self.streak.update(feats.dt, sig["energy"])
            self.arc.update_frame(
                feats.dt, sig["energy"] * (0.55 + 0.45 * self.coherence.score),
                self.streak.heat)
            self.max_streak_tier = max(self.max_streak_tier, self.streak.tier)

        # 2) gestures: recognized + force-fired commands -> actions + flashes
        moves = [g.name for g in self.recognizer.update(frame, feats)]
        moves += list(commands)
        for move in moves:
            self._apply_move(move, frame.t)

        # 3) stillness / presence gate (the phantom fix). An absent body gates
        # instantly; a present body must hold under the floor for the dwell --
        # debounced entry, instant release -- so real dancers dipping through
        # the floor between moves don't flicker the gate (and re-fire the
        # entering-stillness one-shots) twenty times a minute.
        if not present:
            gated = True
            self._still_since = None
        elif sig["energy"] < self.cfg.motion_floor:
            if self._still_since is None:
                self._still_since = frame.t
            gated = (self._still
                     or frame.t - self._still_since >= self.cfg.still_dwell_s)
        else:
            self._still_since = None
            gated = False

        # the band answers your hit: an immediate accent in the pocket.
        if onset is not None and not gated:
            self._accent_answer(frame.t, onset)

        # 4) the grid: exact step times from the LATCHED tempo (not the
        # oscillator's wobble). Steps due this frame fire at their true grid
        # time, so the rendered timing is sample-accurate and metronomic. At
        # each beat the phase servo nudges the grid onto the dancer's accents
        # (one hard snap allowed early in a session).
        beat = False
        if self._next_step_t is None:
            self._next_step_t = frame.t
        while frame.t + 1e-9 >= self._next_step_t:
            step_t = self._next_step_t
            self._abs_step += 1
            self.master_step = self._abs_step % self.cfg.loop_steps
            is_beat = self.master_step % self.cfg.steps_per_beat == 0
            beat = beat or is_beat
            self._on_step(sig, eff, nf, gated, step_t)
            self._next_step_t = step_t + self.latch.beat_s / self.cfg.steps_per_beat
            if is_beat and present:
                self._next_step_t += self.servo.correction(
                    self.latch.beat_s, allow_snap=self._abs_step < 128)
                drift = self.servo.persistent_drift()
                if drift is not None:
                    self.latch.trim(drift)       # the PLL's integral term

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

    def _on_step(self, sig, eff, nf, gated, step_t: float) -> None:
        step = self.master_step
        # deterministic harmony as a function of loop position (looper/builder).
        n = len(self.sub.cfg.harmonic_field)
        self.sub.set_field_index((step * n) // self.cfg.loop_steps)
        beat_s = self.latch.beat_s
        steps_per_bar = self.cfg.steps_per_bar
        bar_step = step % steps_per_bar

        if self._abs_step % steps_per_bar == 0:
            self._on_bar(sig, self._abs_step // steps_per_bar, step_t)

        entering_still = gated and not self._still
        self._still = gated
        # the wind-down: freezing mid-song earns a soft falling run -- the
        # band audibly notices you stopped (then the phantom gate holds).
        if (entering_still and self.cfg.game and self._abs_step > 0
                and self.arc.is_unlocked("lead")):
            self.fx.trigger("FREEZE", step_t)

        # transient windows act at EMISSION so a combo lands instantly, but
        # the committed plans underneath never change mid-bar.
        a = self._abs_step
        filling = a < self._fill_until
        building = a < self._build_until
        dropping = a < self._drop_until
        breaking = a < self._breakdown_until
        self._vel_bias = (4.0 * self.streak.tier if self.cfg.game else 0.0) \
            + (8.0 if a < self._boost_until else 0.0)
        # a recent body accent makes the next beats land harder (the lean).
        if bar_step % self.cfg.steps_per_beat == 0 and self._lean > 0.5:
            self._vel_bias += self._lean
            self._lean *= 0.5

        # swing: odd 16ths sit late by the style's swing (identity feel).
        swing_t = step_t + (self.swing * beat_s / self.cfg.steps_per_beat
                            if bar_step % 2 == 1 else 0.0)
        if len(self.grid_log) < 65536:           # the rhythm-metrics trace
            self.grid_log.append((self._abs_step, swing_t))

        for stem in self.rack:
            stem.active = False
            audible = self.rack.audible(stem)
            events: List[MusicEvent] = []

            # Locked loops replay deterministically (still subject to mute/solo).
            if stem.looping:
                events = stem.replay(step)
            elif gated:
                # PHANTOM GATE: no new onsets while still/absent, except the
                # single sanctioned breath chord at the moment stillness begins
                # (only for stems the song arc has unlocked).
                if (entering_still and stem.name in ("keys", "texture")
                        and (not self.cfg.game
                             or self.arc.is_unlocked(stem.name))):
                    events = stemlib.revoice_pad(stem, self.sub, sig, beat_s,
                                                 dur_s=2.0 * beat_s)
            else:
                if not ((stem.name in ("drums", "bass") and breaking)
                        or (stem.name in ("lead", "keys", "texture") and dropping)):
                    events = self._realize(stem, bar_step, sig, beat_s,
                                           filling, building)
                stem.capture(step, events)

            if events:
                stem.active = True
                stem.level = max(stem.level, max(
                    (e.b for e in events if e.kind == "note_on"), default=0) / 127.0)
            if audible:
                for ev in events:
                    self._emit(ev, swing_t)
                # FIRE+ streak AND a locked groove: the lead earns an ascending
                # ornament run once per bar, on the back half (never spam).
                if (stem.name == "lead" and events and not gated
                        and self.cfg.game and self.streak.tier >= 2
                        and self.coherence.score >= 0.6 and bar_step == 12):
                    self._arp(stem, events[0], beat_s, swing_t)

    # -- bar planning (committed structure; the body modulates within it) ----

    def _on_bar(self, sig, bar_idx: int, step_t: float) -> None:
        # tempo re-locks land only here, as a deliberate musical event.
        prop = self.latch.proposal()
        if prop is not None:
            self.latch.relock(prop)
            self.fx.flashes.append(Flash(f"TEMPO {int(prop)}", (200, 220, 255),
                                         step_t, 0.8, False))
        if self.cfg.game:
            prev_section = self.arc.section
            for s in self.arc.on_bar():
                self.fx.flashes.append(Flash(
                    f"{s.upper()} UNLOCKED", stemlib.STEM_COLOR[s],
                    step_t, 1.2, False))
            self.gold.on_bar(bar_idx)
            self._update_lift(prev_section, step_t)

        # density -> a committed LEVEL per stem for the whole bar (-1 = tacet).
        # The commit reads the BAR-AVERAGED density (what you actually danced
        # last bar), not an instantaneous sample; velocity keeps following the
        # body inside the bar, but the pattern doesn't.
        self._levels = {}
        boosted = self._abs_step < self._boost_until
        for stem in self.rack:
            acc, cnt = self._dens_acc.get(stem.name, (0.0, 0))
            d = acc / cnt if cnt else float(stem.params.density)
            if self.cfg.game:
                if not self.arc.is_unlocked(stem.name):
                    self._levels[stem.name] = -1
                    continue
                d = min(1.0, d * self.arc.density_mul(stem.name))
            lvl = -1 if d < 0.04 else min(3, int(d * 4.0))
            if lvl >= 0 and boosted:
                lvl = min(3, lvl + 1)
            if lvl >= 0 and self.cfg.game:
                lvl = self.coherence.gate_level(lvl)   # the skill gate
            # slew: the arrangement evolves at most one level per bar, so a
            # bucket flapping on its boundary can't teleport the pattern.
            prev = self._prev_levels.get(stem.name)
            if prev is not None and prev >= 0 and lvl >= 0:
                lvl = prev + int(np.sign(lvl - prev))
            self._levels[stem.name] = lvl
        self._prev_levels = dict(self._levels)

        self._dens_acc = {}

        dl, bl = self._levels.get("drums", -1), self._levels.get("bass", -1)
        kl, tl = self._levels.get("keys", -1), self._levels.get("texture", -1)
        ll = self._levels.get("lead", -1)
        self._drum_bar = G.drum_plan(self.style, dl) if dl >= 0 else None
        self._bass_bar = G.bass_plan(self.style, bl) if bl >= 0 else None
        self._keys_bar = dict(G.keys_plan(self.style, kl)) if kl >= 0 else {}
        self._tex_bar = set(G.texture_plan(tl, bar_idx)) if tl >= 0 else set()
        self._lead_on = ll >= 0
        if self._lead_on:
            self.writer.maybe_compose(ll, list(self._contour), bar_idx)

    def _vel(self, base: float, sig: dict) -> int:
        """Pattern accents live in `base`; the body modulates around them, and
        a loose groove dims the whole kit slightly (skill, not punishment)."""
        v = base + 24.0 * (sig["energy"] - 0.5) + 12.0 * (sig["weight"] - 0.5) \
            + self._vel_bias
        if self.cfg.game:
            v *= 0.86 + 0.14 * self.coherence.score
        return int(np.clip(v, 1, 127))

    def _chord_events(self, stem, sig, dur_s: float, base_vel: float
                      ) -> List[MusicEvent]:
        openness = float(np.clip(sig.get("openness", 0.5), 0, 1))
        tones = self.sub.chord_tones(extended=openness > 0.6)
        notes = sorted({self.sub.fold_into_range(
            self.sub.degree_to_midi(d, octave=int(k * (0.3 + 0.7 * openness))),
            stem.role.lo, stem.role.hi) for k, d in enumerate(tones)})
        return [MusicEvent("note_on", stem.channel, nn,
                           self._vel(base_vel, sig), dur=dur_s, tag=stem.name)
                for nn in notes]

    def _realize(self, stem, bar_step: int, sig: dict, beat_s: float,
                 filling: bool, building: bool) -> List[MusicEvent]:
        """This step's committed plan -> events (velocity follows the body)."""
        out: List[MusicEvent] = []
        if stem.name == "drums":
            slots = list(self._drum_bar[bar_step]) if self._drum_bar else []
            if filling:                       # overlay: snare run up + drive
                if bar_step in G.FILL_SNARE:
                    slots.append(("snare",
                                  64 + 10 * G.FILL_SNARE.index(bar_step)))
                if bar_step in G.FILL_HAT and not any(p == "hat" for p, _ in slots):
                    slots.append(("hat", G.LANE_VEL["hat"]))
            if building and bar_step % 2 == 0 \
                    and not any(p == "hat" for p, _ in slots):
                slots.append(("hat", 44 + 2 * bar_step))   # rising 8ths
            for piece, base in slots:
                out.append(MusicEvent("note_on", stem.channel,
                                      stemlib.DRUM_PIECE[piece],
                                      self._vel(base, sig), dur=0.06,
                                      tag="drums"))
            return out

        if stem.name == "bass":
            for off, base, dur_b in (self._bass_bar[bar_step]
                                     if self._bass_bar else []):
                note = self.sub.fold_into_range(
                    self.sub.degree_to_midi(self.sub.chord_root_degree + off),
                    stem.role.lo, stem.role.hi)
                out.append(MusicEvent("note_on", stem.channel, note,
                                      self._vel(base, sig), dur=dur_b * beat_s,
                                      tag="bass"))
            return out

        if stem.name == "keys":
            kind = self._keys_bar.get(bar_step)
            if kind == "pad":
                return self._chord_events(stem, sig, 3.6 * beat_s, 50)
            if kind == "stab":
                return self._chord_events(stem, sig, 0.45 * beat_s, 64)
            return out

        if stem.name == "lead":
            if not self._lead_on:
                return out
            flow = float(sig.get("flow", 0.5))
            for slot, degree, strong in self.writer.plan(
                    self.sub.chord_root_degree):
                if slot != bar_step:
                    continue
                note = self.sub.fold_into_range(
                    self.sub.degree_to_midi(
                        degree, octave=int(stem.params.register_octave)),
                    stem.role.lo, stem.role.hi)
                out.append(MusicEvent(
                    "note_on", stem.channel, note,
                    self._vel(74 + (6 if strong else 0), sig),
                    dur=(0.45 + 0.5 * flow) * beat_s, tag="lead"))
            return out

        # texture: the committed polyrhythm, quiet and long.
        if bar_step in self._tex_bar:
            note = self.sub.snap(float(np.clip(stem.params.pitch, 0, 1)),
                                 stem.role, chord_weighted=True)
            out.append(MusicEvent("note_on", stem.channel, note,
                                  self._vel(40, sig), dur=2.0 * beat_s,
                                  tag="texture"))
        return out

    def _quantize_fx(self, t: float) -> float:
        """Snap a gesture one-shot onto the next 16th so even spontaneous hits
        land in the pocket."""
        return max(t, self._next_step_t) if self._next_step_t is not None else t

    # -- gesture actions ---------------------------------------------------

    def _apply_move(self, move: str, t: float) -> None:
        # Salience cap (see class consts): silently drop a move that's firing
        # constantly -- on real dancers JUMP/PUNCH-like motion can trigger
        # every other beat, which spams fills and drowns the deliberate moves.
        times = self._move_times.setdefault(move, [])
        times[:] = [x for x in times if t - x <= self.MOVE_CAP_SUSTAIN_WINDOW]
        burst = sum(1 for x in times if t - x <= self.MOVE_CAP_WINDOW)
        if burst >= self.MOVE_CAP or len(times) >= self.MOVE_CAP_SUSTAIN:
            return
        times.append(t)

        # Always flash so the stage can show any recognized/forced move.
        color, big = STYLE.get(move, ((230, 230, 230), False))
        self.fx.flashes.append(Flash(move, color, t, 0.5 if big else 0.4, big))

        if self.cfg.game:
            self.streak.observe_move()
            if self.gold.observe(move, t):
                self._gold_hit(t)
            combo = self.combos.observe(move, t, 60.0 / max(self.bpm, 60.0))
            if combo is not None:
                self._apply_combo(combo, t)

        for g in self.resolver.resolve_gesture(move):
            self._do_action(g, t)

    # -- game payoffs --------------------------------------------------------

    def _gold_hit(self, t: float) -> None:
        """A gold-move challenge landed inside its window: PERFECT."""
        self.streak.observe_move(big=True)
        self.arc.bump(0.12)
        self._build_until = max(self._build_until,
                                self._abs_step + self.cfg.steps_per_bar)
        self.fx.trigger("HANDS UP", t)             # the reward riser
        # appended last so the big banner reads PERFECT!, not the fx name.
        self.fx.flashes.append(Flash("PERFECT!", GOLD_COLOR, t, 0.9, True))

    def _apply_combo(self, combo, t: float) -> None:
        """A recognised move sequence: a big named payoff, same every time."""
        self.combo_log.append((t, combo.name))
        self.streak.observe_move(big=True)
        self.arc.bump(0.15)
        bar = self.cfg.steps_per_bar
        a = self._abs_step
        p = combo.payoff
        if p == "supernova":                       # drop a bar, then the slam
            self._drop_until = max(self._drop_until, a + bar)
            self._boost_until = max(self._boost_until, a + 3 * bar)
            self.fx.trigger("HANDS UP", t)
        elif p == "earthquake":                    # bass an octave down, driven
            self._quake_until = max(self._quake_until, a + 2 * bar)
            self._fill_until = max(self._fill_until, a + bar)
            self.fx.trigger("SQUAT", t)
        elif p == "clap_storm":
            self._fill_until = max(self._fill_until, a + 2 * bar)
            self.fx.trigger("CLAP", t)
        elif p == "wave":                          # the long rising build
            self._build_until = max(self._build_until, a + 2 * bar)
            self.fx.trigger("HANDS UP", t)
        elif p == "eclipse":                       # flip light <-> dark
            self._breakdown_until = max(self._breakdown_until, a + bar)
            self._eclipse_flip(t)
        elif p == "knockout":                      # three ascending chord stabs
            self.fx.trigger("KNOCKOUT", t)
            self._fill_until = max(self._fill_until, a + bar)
            self._boost_until = max(self._boost_until, a + 2 * bar)
        # appended last so the big banner reads the combo's name.
        self.fx.flashes.append(Flash(combo.name, combo.color, t, 0.9, True))

    def _next_beat_t(self) -> Optional[float]:
        """Wall time of the next grid BEAT (None before the grid starts)."""
        if self._next_step_t is None:
            return None
        spb = self.cfg.steps_per_beat
        step_s = self.latch.beat_s / spb
        return self._next_step_t + ((-(self._abs_step + 1)) % spb) * step_s

    def _accent_answer(self, t: float, strength: float) -> None:
        """A STRONG movement (well above this dancer's own norm) gets an
        immediate drum answer, snapped to the NEAREST 16th (<= ~60 ms shift:
        in the pocket, still unmistakably yours). Rare by design -- a
        punctuation, not a second drummer -- and tagged 'accent' so the
        committed-groove metrics measure the pattern, not the punctuation."""
        if not self.cfg.game or not self.arc.is_unlocked("drums"):
            return
        if strength < 0.6 or t - self._accent_last_t < 1.5 * self.latch.beat_s:
            return
        if not self.rack.audible(self.rack.get("drums")):
            return
        self._accent_last_t = t
        tq = t
        if self._next_step_t is not None:
            step_s = self.latch.beat_s / self.cfg.steps_per_beat
            prev_step = self._next_step_t - step_s
            tq = prev_step if t - prev_step < self._next_step_t - t \
                else self._next_step_t
        piece = "snare" if strength > 0.75 else "hat"
        vel = int(np.clip(56 + 52 * strength, 1, 120))
        self._emit(MusicEvent("note_on", stemlib.CHANNELS["drums"],
                              stemlib.DRUM_PIECE[piece], vel, dur=0.08,
                              tag="accent"), tq)

    def _request_fill(self, t: float) -> None:
        """The fill budget: a drummer fills about once a phrase, not every
        other beat. Over-budget requests still answer the move instantly with
        a single accent hit, so the causation stays legible."""
        bar = self.cfg.steps_per_bar
        if self._abs_step >= self._fill_until + 3 * bar:
            self._fill_until = self._abs_step + bar
        else:
            self.backend.send(MusicEvent(
                "note_on", stemlib.CHANNELS["drums"], stemlib.DRUM_PIECE["hat"],
                96, t=self._quantize_fx(t), dur=0.1, tag="drums"))

    def _log_key(self, t: float) -> None:
        self.key_log.append((t, self.sub.cfg.tonic, self.sub.cfg.scale))

    def _eclipse_flip(self, t: float) -> None:
        """Toggle between this person's home scale and the mapping's dark mood."""
        dark = self.mapping.moods.get("dark", "phrygian")
        to = dark if self.sub.cfg.scale != dark else self._base_scale
        if to in SCALES:
            self.sub.cfg.scale = to
            self.sub._scale = SCALES[to]
            self.mood = to
            self._log_key(t)

    def _arp(self, stem, base_ev, beat_s: float, t0: float) -> None:
        """Two extra chord-tone steps above the lead note, staggered on the
        16th grid (scheduled sends, like fx one-shots, so loops stay clean)."""
        step_s = beat_s / self.cfg.steps_per_beat
        root = self.sub.chord_root_degree
        for k in (1, 2):
            note = self.sub.fold_into_range(
                self.sub.degree_to_midi(root + 2 * k + 2,
                                        octave=int(stem.params.register_octave)),
                stem.role.lo, stem.role.hi)
            t = t0 + k * step_s
            vel = max(40, int(base_ev.b) - 12 * k)
            self.backend.send(MusicEvent("note_on", stem.channel, note, vel,
                                         t=t, dur=0.12, tag="lead"))
            self.backend.send(MusicEvent("note_off", stem.channel, note, 0,
                                         t=t + 0.12, tag="lead"))

    def _update_lift(self, prev_section: str, t: float) -> None:
        """THE LIFT: reaching PEAK transposes the whole world up two semitones
        (the classic modulation); cooling back down undoes it. Locked loop
        buffers are transposed with it, so loops recorded before the lift stay
        in key with everything else."""
        now = self.arc.section
        if now == prev_section:
            return
        if now == "peak" and not self._lifted:
            self._transpose(+2, t)
            self._lifted = True
            self.fx.flashes.append(Flash("THE LIFT", GOLD_COLOR, t,
                                         0.8, False))
        elif prev_section == "peak" and self._lifted:
            self._transpose(-2, t)
            self._lifted = False

    def _transpose(self, semis: int, t: float) -> None:
        self.sub.cfg.tonic += semis
        self._log_key(t)
        for stem in self.rack:
            if stem.channel == 10:
                continue
            for slot in stem.buffer:
                for ev in slot:
                    if ev.kind in ("note_on", "note_off"):
                        ev.a = int(np.clip(ev.a + semis, 0, 127))

    def _do_action(self, g, t: float) -> None:
        action, target, params = g.action, g.target, g.params
        if action == "fx":
            # One-shot hit on the target stem; EffectEngine tags "fx" + carries
            # its own channel. Map the binding's params to a known fx move.
            self.fx.trigger(_fx_move(params), t)
        elif action == "fill":
            self._request_fill(t)
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
                self._log_key(t)
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
                onsets=stem.loop_onsets(), timbre=stem.timbre or stem.params.timbre,
                locked=bool(self.cfg.game and not self.arc.is_unlocked(stem.name))))
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
            mood=self.mood, beat=bool(beat), game=self._game_ui(),
            still=bool(gated))

    def _game_ui(self) -> Optional[GameUI]:
        if not self.cfg.game:
            return None
        # progress through the current bar drives the challenge countdown.
        spb = self.cfg.steps_per_bar
        bar_frac = (max(self._abs_step, 0) % spb) / float(spb)
        ident = self.identity
        return GameUI(
            section=self.arc.section, sections=list(SECTIONS),
            heat=float(self.arc.heat), streak=float(self.streak.heat),
            streak_tier=self.streak.tier_name,
            challenge_move=self.gold.move, challenge_state=self.gold.state,
            challenge_frac=(bar_frac if self.gold.state in ("announce", "window")
                            else 0.0),
            unlocked={s.name: self.arc.is_unlocked(s.name) for s in self.rack},
            identity_name=ident.name if ident else "",
            identity_tagline=ident.tagline if ident else "",
            identity_color=ident.color if ident else (220, 220, 235),
            groove=float(self.coherence.score))


def _fx_move(params: dict) -> str:
    """Map an fx binding's params to a move name EffectEngine knows."""
    hit = (params or {}).get("hit", "")
    return {"snare": "CLAP", "stab": "PUNCH"}.get(hit, "CLAP")
