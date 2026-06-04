"""Real-time audio backend for the live sandbox -- zero external setup.

Synthesizes MusicEvents straight to the speakers via `sounddevice`, reusing the
shared voices, so you can dance and hear the song build with no DAW / MIDI port /
soundfont. Degrades gracefully: if `sounddevice` isn't installed it becomes a
silent no-op (and tells you), so the visual sandbox still runs.

A note_on with a known `dur` is rendered to a finite buffer immediately and mixed
in the audio callback; note_off/cc are ignored (notes are short and self-ending).
"""

from __future__ import annotations

import threading
from typing import List, Tuple

import numpy as np

from .output import Backend, MusicEvent
from .voices import SR, render_note

BLOCK = 512
MAX_VOICES = 48


class RealtimeSynth(Backend):
    def __init__(self, samplerate: int = SR, master: float = 0.6,
                 block: int = BLOCK):
        self.sr = samplerate
        self.master = master
        self.ok = False
        self.error = ""
        self._lock = threading.Lock()
        self._voices: List[List] = []          # [buf, pos] pairs
        self._stream = None
        try:
            import sounddevice as sd            # lazy: optional dependency
            self._sd = sd
            self._stream = sd.OutputStream(
                samplerate=samplerate, channels=1, blocksize=block,
                dtype="float32", callback=self._callback)
            self._stream.start()
            self.ok = True
        except Exception as e:                  # no portaudio / no device / etc.
            self.error = f"{type(e).__name__}: {e}"

    # -- Backend interface --------------------------------------------------

    def send(self, ev: MusicEvent) -> None:
        if not self.ok or ev.kind != "note_on" or not ev.dur:
            return
        sig = render_note(ev.channel, ev.a, ev.dur, ev.b).astype(np.float32)
        with self._lock:
            if len(self._voices) >= MAX_VOICES:
                self._voices.pop(0)
            self._voices.append([sig, 0])

    def panic(self) -> None:
        with self._lock:
            self._voices.clear()

    def close(self) -> None:  # pragma: no cover - hardware
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass

    # -- audio thread -------------------------------------------------------

    def _callback(self, outdata, frames, time_info, status):  # pragma: no cover
        mix = self._mix(frames)
        outdata[:, 0] = mix

    def _mix(self, frames: int) -> np.ndarray:
        """Pure mixing step (separated so it's unit-testable without a device)."""
        out = np.zeros(frames, dtype=np.float32)
        with self._lock:
            keep = []
            for v in self._voices:
                buf, pos = v
                chunk = buf[pos:pos + frames]
                out[:len(chunk)] += chunk
                v[1] = pos + frames
                if v[1] < len(buf):
                    keep.append(v)
            self._voices = keep
        out *= self.master
        return np.tanh(out).astype(np.float32)     # soft clip
