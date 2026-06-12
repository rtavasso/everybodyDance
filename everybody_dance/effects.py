"""Effects layer: a recognised gesture -> a one-shot musical hit + a screen flash.

These fire ON TOP of whatever continuous engine is running, so open-ended dancing
keeps shaping the music while specific moves punch in salient, repeatable events
(a clap snare, a hands-up riser, a T-pose drop). Notes are snapped to the active
scale via the Substrate, so hits stay musical. Fully deterministic.

Each note is emitted as a note_on (with `dur`, for the real-time synth) plus a
future-timestamped note_off (for the offline WAV renderer) so both backends work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .output import Backend, MusicEvent
from .substrate import Substrate

# BGR colour + whether it's a full-screen "gold move" style flash
STYLE: Dict[str, Tuple[Tuple[int, int, int], bool]] = {
    "CLAP": ((255, 255, 255), False),
    "PUNCH": ((90, 90, 255), False),
    "HANDS UP": ((90, 240, 255), True),
    "T-POSE": ((255, 120, 220), True),
    "SQUAT": ((255, 200, 80), False),
    "ARMS CROSSED": ((200, 160, 255), False),
    "RAISE LEFT": ((120, 255, 160), False),
    "RAISE RIGHT": ((160, 220, 255), False),
    "KNOCKOUT": ((90, 90, 255), True),
    "FREEZE": ((255, 220, 170), False),
}
DEFAULT_STYLE = ((230, 230, 230), False)


@dataclass
class Flash:
    name: str
    color: Tuple[int, int, int]
    t0: float
    ttl: float = 0.45
    big: bool = False

    def alpha(self, t: float) -> float:
        # clamped both ways: a quantized fx flash may carry a t0 slightly in
        # the future (it fires on the next grid step).
        return min(1.0, max(0.0, 1.0 - (t - self.t0) / self.ttl))


class EffectEngine:
    def __init__(self, backend: Backend, substrate: Substrate):
        self.backend = backend
        self.sub = substrate
        self.flashes: List[Flash] = []
        self.log: List[Tuple[float, str]] = []
        self.quantize = None     # optional t -> t hook (studio snaps to grid)

    # -- public ------------------------------------------------------------

    def trigger(self, name: str, t: float) -> None:
        if self.quantize is not None:
            t = float(self.quantize(t))      # one-shots land in the pocket
        getattr(self, "_fx_" + name.lower().replace(" ", "_").replace("-", "_"),
                self._fx_default)(t)
        color, big = STYLE.get(name, DEFAULT_STYLE)
        # one flash per moment per name: the caller may already have flashed
        # this move (studio does); a duplicate is visual noise.
        if not any(f.name == name and abs(f.t0 - t) < 0.35
                   for f in self.flashes):
            self.flashes.append(Flash(name, color, t, 0.5 if big else 0.4, big))
        self.log.append((t, name))

    def active_flashes(self, t: float) -> List[Flash]:
        self.flashes = [f for f in self.flashes if f.alpha(t) > 0]
        return self.flashes

    # -- note helpers ------------------------------------------------------

    def _note(self, ch, pitch, vel, dur, t):
        self.backend.send(MusicEvent("note_on", ch, int(pitch), int(vel),
                                     dur=dur, t=t, tag="fx"))
        self.backend.send(MusicEvent("note_off", ch, int(pitch), 0,
                                     t=t + dur, tag="fx"))

    def _snap(self, v01, role):
        r = self.sub.cfg.roles[role]
        return self.sub.snap(float(min(max(v01, 0), 1)), r, chord_weighted=True)

    def _run(self, role, ch, vals, t, step=0.06, dur=0.18, vel=96):
        for k, v in enumerate(vals):
            self._note(ch, self._snap(v, role), vel, dur, t + k * step)

    # -- one-shot effects (deterministic) ----------------------------------

    def _fx_clap(self, t):       # hand-clap / snare
        self._note(10, 39, 118, 0.12, t)

    def _fx_raise_left(self, t):     # single high lead accent
        self._note(3, self._snap(0.85, "lead"), 104, 0.3, t)

    def _fx_raise_right(self, t):    # single mid lead accent + tom
        self._note(3, self._snap(0.6, "lead"), 104, 0.3, t)
        self._note(10, 45, 90, 0.18, t)

    def _fx_punch(self, t):      # chord stab (keys)
        for v in (0.4, 0.55, 0.7):
            self._note(2, self._snap(v, "chord"), 110, 0.22, t)

    def _fx_hands_up(self, t):   # ascending riser (lead)
        self._run("lead", 3, [0.3, 0.5, 0.7, 0.9, 1.0], t, step=0.07, dur=0.16)

    def _fx_arms_crossed(self, t):   # descending downlifter (lead)
        self._run("lead", 3, [1.0, 0.8, 0.6, 0.4, 0.2], t, step=0.07, dur=0.16)

    def _fx_t_pose(self, t):     # the drop: low bass + sustained chord
        self._note(1, self._snap(0.1, "bass"), 120, 1.4, t)
        for v in (0.45, 0.6, 0.78):
            self._note(2, self._snap(v, "chord"), 100, 1.4, t)

    def _fx_squat(self, t):      # bass drop
        self._note(1, self._snap(0.05, "bass"), 122, 0.8, t)

    def _fx_knockout(self, t):   # three ascending chord stabs (the combo K.O.)
        for k, base in enumerate((0.35, 0.55, 0.75)):
            for v in (base, base + 0.1, base + 0.2):
                self._note(2, self._snap(v, "chord"), 112, 0.2, t + 0.14 * k)

    def _fx_freeze(self, t):     # the wind-down: a soft falling run as you freeze
        self._run("lead", 3, [0.7, 0.5, 0.3], t, step=0.07, dur=0.14, vel=58)

    def _fx_default(self, t):
        self._note(2, self._snap(0.5, "chord"), 100, 0.2, t)
