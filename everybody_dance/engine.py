"""The engine -- one process: pose -> features + Effort -> entrained clock ->
substrate readout -> MIDI/OSC. Runs at 30-60 fps.

Two phases:
  * calibrate -- ~20 s of free movement builds the per-person Profile while the
    oscillator is already learning the characteristic tempo;
  * play -- the instrument.

A tiny scheduler turns note_on events with a `dur` into matching note_offs.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .features import FeatureExtractor
from .laban import LabanEstimator
from .oscillator import EntrainedClock
from .output import Backend, MusicEvent
from .personalization import Calibrator, Normalizer, Profile
from .pose import PoseSource
from .readout import Readout
from .substrate import Substrate, SubstrateConfig, biased_config


@dataclass
class EngineConfig:
    fps: float = 30.0
    tempo_mode: str = "entrain"      # "entrain" | "fixed"
    coupling: float = 0.4            # the most important dial in the system
    fixed_hz: float = 2.0
    calibrate_s: float = 20.0
    latency_compensation_s: float = 0.0
    seed: int = 0
    # Exhibit option: go quiet when the body is still, so the music reads as
    # caused (vs. the default ambient bed that sustains through stillness).
    motion_gate: bool = False
    gate_threshold: float = 0.12     # normalised energy below this counts as "still"
    gate_release_s: float = 0.35     # stillness this long -> stop emitting new notes


@dataclass
class FrameTrace:
    """What happened on one frame -- handy for logging / the headless demo."""
    t: float
    bpm: float
    phase: float
    confidence: float
    locked: bool
    energy: float
    effort: dict
    n_events: int
    freeze: float


class Engine:
    def __init__(self, source: PoseSource, backend: Backend,
                 cfg: Optional[EngineConfig] = None,
                 profile: Optional[Profile] = None):
        self.source = source
        self.backend = backend
        self.cfg = cfg or EngineConfig()
        self.features = FeatureExtractor(fps=self.cfg.fps)
        self.laban = LabanEstimator()
        self.clock = EntrainedClock(
            tempo_mode=self.cfg.tempo_mode,
            coupling=self.cfg.coupling,
            fixed_hz=self.cfg.fixed_hz,
            steps_per_beat=SubstrateConfig.default().steps_per_beat,
        )
        self.profile = profile
        self.normalizer: Optional[Normalizer] = None
        self.substrate: Optional[Substrate] = None
        self.readout: Optional[Readout] = None
        self._offs: List[Tuple[float, MusicEvent]] = []  # heap of (t_off, ev)
        self.traces: List[FrameTrace] = []
        self._still_s = 0.0              # accrued stillness, for the motion gate
        self._gated = False              # current motion-gate state

    # -- lifecycle ---------------------------------------------------------

    def set_coupling(self, c: float) -> None:
        self.cfg.coupling = c
        self.clock.set_coupling(c)

    def _activate(self, profile: Profile) -> None:
        self.profile = profile
        self.normalizer = Normalizer(profile)
        cfg = biased_config(profile.signature)
        self.substrate = Substrate(cfg)
        self.clock.steps_per_beat = cfg.steps_per_beat
        self.readout = Readout(self.substrate, seed=self.cfg.seed,
                               latency_compensation_s=self.cfg.latency_compensation_s)

    # -- core step ---------------------------------------------------------

    def _process(self, frame, calibrator: Optional[Calibrator]) -> Optional[FrameTrace]:
        feats = self.features.update(frame)
        drivers = self.laban.update(feats, feats.dt)
        # The clock entrains to the vertical COM bounce -- a clean fundamental.
        clk = self.clock.update(float(feats.bounce), feats.dt)

        if calibrator is not None:
            calibrator.observe(feats, drivers, clk.tempo_hz)
            return None

        eff = self.normalizer.effort(drivers)
        nf = self.normalizer.features(feats, eff)
        events = self.readout.step(feats, eff, nf, clk)
        if self.cfg.motion_gate:
            # while the body is still: emit no new notes, and on the falling edge
            # release whatever is still sounding so stillness actually goes quiet
            # (not just onset-quiet -- the sustained pads must stop too).
            self._still_s = (self._still_s + feats.dt
                             if nf.energy < self.cfg.gate_threshold else 0.0)
            gated = self._still_s >= self.cfg.gate_release_s
            if gated:
                if not self._gated:
                    self._release_all(feats.t)
                events = [e for e in events if e.kind != "note_on"]
            self._gated = gated
        self._emit(events, feats.t)
        self._flush_offs(feats.t)

        return FrameTrace(
            t=feats.t, bpm=clk.bpm, phase=clk.phase, confidence=clk.confidence,
            locked=clk.locked, energy=nf.energy, effort=eff.as_dict(),
            n_events=len([e for e in events if e.kind == "note_on"]),
            freeze=feats.events.get("freeze", 0.0),
        )

    def _emit(self, events: List[MusicEvent], t: float) -> None:
        for ev in events:
            ev.t = t
            self.backend.send(ev)
            if ev.kind == "note_on" and ev.dur:
                off = MusicEvent("note_off", ev.channel, ev.a, 0, t=t + ev.dur,
                                 tag=ev.tag)
                heapq.heappush(self._offs, (t + ev.dur, _Seq.next(), off))

    def _flush_offs(self, t: float) -> None:
        while self._offs and self._offs[0][0] <= t:
            _, _, off = heapq.heappop(self._offs)
            self.backend.send(off)

    def _release_all(self, t: float) -> None:
        """Send every pending note-off now (motion gate closing -> go quiet)."""
        while self._offs:
            _, _, off = heapq.heappop(self._offs)
            off.t = t
            self.backend.send(off)

    # -- run ---------------------------------------------------------------

    def run(self) -> List[FrameTrace]:
        """Run calibration (if no profile) then play, to exhaustion of source."""
        frames = self.source.frames()
        calibrator = None
        start = None

        if self.profile is None:
            calibrator = Calibrator()

        for frame in frames:
            if start is None:
                start = frame.t
            in_calib = calibrator is not None and (frame.t - start) < self.cfg.calibrate_s

            if calibrator is not None and not in_calib:
                # transition: finalise profile, activate the instrument
                sig = self.features.signature.summary()
                prof = calibrator.finalize(signature=sig)
                self._activate(prof)
                calibrator = None

            if self.readout is None and calibrator is None:
                # profile supplied up front
                self._activate(self.profile)

            trace = self._process(frame, calibrator if in_calib else None)
            if trace is not None:
                self.traces.append(trace)

        # Never leave notes hanging.
        self.backend.panic()
        return self.traces


class _Seq:
    _n = 0

    @classmethod
    def next(cls) -> int:
        cls._n += 1
        return cls._n
