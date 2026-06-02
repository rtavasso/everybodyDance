"""Control input -- the out-of-band 'pedal' that commits a loop and advances.

A foot pedal almost always emulates a keypress, so a keyboard listener covers
both. For headless runs and tests we use a scripted control track (commands at
fixed times), so the whole loop-building performance can be rendered and
asserted without a pedal or a camera.
"""

from __future__ import annotations

import queue
import sys
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


class Command(Enum):
    COMMIT = "commit"     # lock the current loop, advance to the next instrument
    UNDO = "undo"         # clear/redo the current (or last) track
    RESET = "reset"       # wipe everything, back to the first track
    QUIT = "quit"


class ControlInput:
    def poll(self, t: float) -> List[Command]:
        """Return any commands that have arrived by time `t` (seconds)."""
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover
        pass


class ScriptedControl(ControlInput):
    """Commands at scheduled times -- for headless renders and tests."""

    def __init__(self, schedule: List[Tuple[float, Command]]):
        self._sched = sorted(schedule, key=lambda x: x[0])
        self._i = 0

    def poll(self, t: float) -> List[Command]:
        out = []
        while self._i < len(self._sched) and self._sched[self._i][0] <= t:
            out.append(self._sched[self._i][1])
            self._i += 1
        return out


_KEYMAP = {" ": Command.COMMIT, "\n": Command.COMMIT, "\r": Command.COMMIT,
           "c": Command.COMMIT, "u": Command.UNDO, "r": Command.RESET,
           "q": Command.QUIT}


class KeyboardControl(ControlInput):  # pragma: no cover - interactive only
    """Non-blocking keyboard listener (space/enter/c = commit, u = undo,
    r = reset, q = quit). A foot pedal mapped to a key works transparently."""

    def __init__(self):
        self._q: "queue.Queue[Command]" = queue.Queue()
        import threading
        self._stop = False
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        try:
            for line in iter(sys.stdin.readline, ""):
                if self._stop:
                    break
                for ch in (line.strip().lower() or "\n"):
                    cmd = _KEYMAP.get(ch)
                    if cmd:
                        self._q.put(cmd)
        except Exception:
            pass

    def poll(self, t: float) -> List[Command]:
        out = []
        try:
            while True:
                out.append(self._q.get_nowait())
        except queue.Empty:
            pass
        return out

    def close(self) -> None:
        self._stop = True
