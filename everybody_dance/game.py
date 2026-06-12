"""The game layer -- what makes the Studio *feel like Just Dance*.

Four deterministic mechanics, layered over the continuous mapping + gestures:

  * **Combos** (:class:`ComboTracker`): named move *sequences* inside a beat
    window fire big, repeatable musical payoffs (SQUAT then JUMP = SUPERNOVA:
    a drop into a doubled-density slam). Combos are discoverable, learnable,
    and bigger than any single move -- the skill curve of the instrument.
  * **Streak** (:class:`StreakMeter`): recognised moves heat a meter that
    decays in stillness; tiers (WARM / FIRE / GOLD) push velocity and the
    song arc, so committing to the dance *audibly* raises the stakes.
  * **Song arc** (:class:`SongArc`): a heat-driven section machine
    (intro -> groove -> build -> peak) that *unlocks stems* as the dancer
    earns them. The song starts as drums+bass and grows into the full band;
    every unlock is announced. Unlocks are sticky -- cooling off softens the
    arrangement but never takes an instrument away (a gallery must never
    punish), and section changes land only on bar lines so they stay musical.
  * **Gold moves** (:class:`GoldMoveGame`): the Just Dance signature --
    periodically a specific move is announced, then a timed window opens; hit
    it for a PERFECT payoff. This is also the zero-instruction tutorial: the
    screen teaches the vocabulary one move at a time.

No RNG anywhere: combos/streak/sections/gold scheduling are pure functions of
the dance and the bar counter, so the same session replays byte-identically
and a player can reproduce everything they discover.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# -- combos ------------------------------------------------------------------


@dataclass(frozen=True)
class ComboDef:
    """A named move sequence: fire `seq` in order within `beats` beats."""
    name: str
    seq: Tuple[str, ...]
    beats: float
    color: Tuple[int, int, int]          # BGR flash colour
    payoff: str                          # studio-interpreted payoff key


COMBOS: List[ComboDef] = [
    ComboDef("SUPERNOVA", ("SQUAT", "JUMP"), 4.0, (80, 200, 255), "supernova"),
    ComboDef("EARTHQUAKE", ("STOMP", "STOMP"), 4.0, (60, 120, 255), "earthquake"),
    ComboDef("CLAP STORM", ("CLAP", "CLAP", "CLAP"), 6.0, (255, 255, 255),
             "clap_storm"),
    ComboDef("THE WAVE", ("RAISE LEFT", "RAISE RIGHT", "HANDS UP"), 8.0,
             (120, 255, 200), "wave"),
    ComboDef("ECLIPSE", ("T-POSE", "ARMS CROSSED"), 6.0, (220, 130, 255),
             "eclipse"),
    ComboDef("KNOCKOUT", ("PUNCH", "PUNCH", "PUNCH"), 6.0, (90, 90, 255),
             "knockout"),
]


class ComboTracker:
    """Matches the recent move history against the combo library.

    Longest sequences win (PUNCH x3 fires KNOCKOUT, not a 2-long subset), and a
    fired combo clears the history so one flurry can't cascade into several."""

    def __init__(self, combos: Optional[List[ComboDef]] = None):
        self.combos = sorted(list(combos or COMBOS), key=lambda c: -len(c.seq))
        self._hist: List[Tuple[float, str]] = []

    def observe(self, move: str, t: float, beat_s: float) -> Optional[ComboDef]:
        self._hist.append((t, move))
        self._hist = [(ht, hm) for ht, hm in self._hist
                      if t - ht <= 16.0 * beat_s][-12:]
        for combo in self.combos:
            k = len(combo.seq)
            if (len(self._hist) >= k
                    and tuple(m for _, m in self._hist[-k:]) == combo.seq
                    and t - self._hist[-k][0] <= combo.beats * beat_s):
                self._hist.clear()
                return combo
        return None


# -- streak ------------------------------------------------------------------


class StreakMeter:
    """Heat from recognised moves, sustained by energy, decaying in stillness."""

    TIERS = ["", "WARM", "FIRE", "GOLD"]
    _EDGES = [0.25, 0.5, 0.8]            # heat thresholds for tiers 1..3

    def __init__(self):
        self.heat = 0.0

    def observe_move(self, big: bool = False) -> None:
        self.heat = float(min(1.0, self.heat + (0.3 if big else 0.18)))

    def update(self, dt: float, energy: float) -> None:
        # High energy roughly holds the meter; stillness drains it in ~10 s.
        self.heat = float(np.clip(
            self.heat + dt * (0.06 * float(energy) - 0.08), 0.0, 1.0))

    @property
    def tier(self) -> int:
        return int(sum(self.heat >= e for e in self._EDGES))

    @property
    def tier_name(self) -> str:
        return self.TIERS[self.tier]


# -- song arc ----------------------------------------------------------------

SECTIONS = ["intro", "groove", "build", "peak"]

# What each section *adds* to the unlocked set when first reached.
SECTION_UNLOCKS: Dict[str, Tuple[str, ...]] = {
    "intro": ("drums", "bass"), "groove": ("keys",), "build": ("lead",),
    "peak": ("texture",)}

# Heat needed (at a bar line) to rise into a section. Tuned so an energetic
# dancer earns the full band within a handful of bars, and the GOLD push
# (moves/combos) is what reaches the peak.
SECTION_HEAT = {"groove": 0.15, "build": 0.32, "peak": 0.55}

# Per-section, per-stem density multipliers (default 1.0): the arrangement
# breathes with the arc instead of every stem flatlining at its mapped density.
_DENSITY_MUL: Dict[str, Dict[str, float]] = {
    "intro": {"drums": 0.9, "bass": 0.8},
    "groove": {},
    "build": {"drums": 1.1, "keys": 1.05},
    "peak": {"drums": 1.2, "bass": 1.1, "keys": 1.1, "lead": 1.15},
}


class SongArc:
    """Heat-driven section machine with sticky stem unlocks.

    `update_frame` integrates dance heat every frame; `on_bar` (call exactly on
    bar lines) is the only place sections change, so transitions stay musical.
    Returns newly unlocked stem names so the caller can announce them."""

    def __init__(self):
        self.section = "intro"
        self.heat = 0.0
        self.unlocked = set(SECTION_UNLOCKS["intro"])
        self.bars_in_section = 0
        self.sections_visited = ["intro"]

    def update_frame(self, dt: float, energy: float, streak_level: float) -> None:
        target = float(np.clip(0.75 * float(energy) + 0.4 * float(streak_level),
                               0.0, 1.0))
        k = float(np.clip(dt / 1.2, 0.0, 1.0))        # ~1.2 s time constant
        self.heat += k * (target - self.heat)

    def bump(self, amount: float) -> None:
        """A discrete shove (combo / gold hit) toward the next section."""
        self.heat = float(np.clip(self.heat + amount, 0.0, 1.0))

    def on_bar(self) -> List[str]:
        self.bars_in_section += 1
        idx = SECTIONS.index(self.section)
        new: List[str] = []
        if idx < len(SECTIONS) - 1 and self.heat >= SECTION_HEAT[SECTIONS[idx + 1]]:
            self.section = SECTIONS[idx + 1]
            self.bars_in_section = 0
            if self.section not in self.sections_visited:
                self.sections_visited.append(self.section)
            for s in SECTION_UNLOCKS[self.section]:
                if s not in self.unlocked:
                    self.unlocked.add(s)
                    new.append(s)
        elif (idx > 0 and self.bars_in_section >= 2
                and self.heat < 0.5 * SECTION_HEAT[SECTIONS[idx]]):
            self.section = SECTIONS[idx - 1]          # cool off (stems stay)
            self.bars_in_section = 0
        return new

    def density_mul(self, stem: str) -> float:
        return _DENSITY_MUL[self.section].get(stem, 1.0)

    def is_unlocked(self, stem: str) -> bool:
        return stem in self.unlocked


# -- gold moves ----------------------------------------------------------------

# The challenge rotation: the moves that read best from across a room, in a
# fixed teaching order (also the implicit tutorial sequence).
GOLD_ROTATION = ["JUMP", "CLAP", "HANDS UP", "SQUAT", "PUNCH", "T-POSE",
                 "STOMP", "RAISE LEFT"]
GOLD_COLOR = (60, 215, 255)              # BGR gold


class GoldMoveGame:
    """Periodic timed move challenges: announce one bar, window one bar.

    Schedule is a pure function of the bar counter (first prompt at bar
    `start_bar`, then every `period_bars`); outcomes depend only on the moves
    the dancer actually performs. Misses are silent -- no punishment."""

    def __init__(self, start_bar: int = 4, period_bars: int = 8):
        self.start_bar = start_bar
        self.period_bars = period_bars
        self.state = "idle"                  # idle | announce | window
        self.move: Optional[str] = None
        self.prompts = 0
        self.hits = 0
        self.last_hit_t: Optional[float] = None
        self._k = 0

    def on_bar(self, bar: int) -> None:
        if self.state == "announce":
            self.state = "window"
        elif self.state == "window":
            self.state = "idle"              # window closed unhit: silent miss
            self.move = None
        if (self.state == "idle" and bar >= self.start_bar
                and (bar - self.start_bar) % self.period_bars == 0):
            self.move = GOLD_ROTATION[self._k % len(GOLD_ROTATION)]
            self._k += 1
            self.prompts += 1
            self.state = "announce"

    def observe(self, move: str, t: float) -> bool:
        if self.state == "window" and move == self.move:
            self.state = "idle"
            self.move = None
            self.hits += 1
            self.last_hit_t = t
            return True
        return False


# -- the per-frame UI snapshot -------------------------------------------------


@dataclass
class GameUI:
    """Everything the stage needs to draw the game state, plain data."""
    section: str
    sections: List[str]
    heat: float
    streak: float
    streak_tier: str
    challenge_move: Optional[str]        # None when no challenge running
    challenge_state: str                 # "idle" | "announce" | "window"
    challenge_frac: float                # 0..1 progress through the phase
    unlocked: Dict[str, bool]
    identity_name: str = ""
    identity_tagline: str = ""
    identity_color: Tuple[int, int, int] = (220, 220, 235)
