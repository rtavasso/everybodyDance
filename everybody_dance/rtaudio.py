"""Real-time audio backend for the live sandbox -- zero external setup.

Synthesizes MusicEvents straight to the speakers via `sounddevice`, so you can
dance and hear the song build with no DAW / MIDI port / soundfont. Degrades
gracefully: if `sounddevice` isn't installed it becomes a silent no-op (and
tells you), so the visual sandbox still runs.

A note_on with a known `dur` is rendered to a finite buffer immediately and mixed
in the audio callback; note_off/cc are ignored (notes are short and self-ending).

Two voice paths:
  * the simple shared :mod:`voices` (the legacy default, very cheap), and
  * the rich :mod:`timbre` presets, when a tag/channel -> preset map is set via
    ``set_preset`` -- this is how the live Studio sounds like its offline
    render. Rendered notes are cached on quantised (preset, pitch, dur, vel)
    keys, so each timbre/pitch costs its synthesis once and is free after.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Tuple

import numpy as np

from .output import Backend, MusicEvent
from .voices import SR, render_note

BLOCK = 512
MAX_VOICES = 48
CACHE_MAX = 768          # rendered-note cache entries (a few minutes of variety)

# Drum pitch -> percussive preset (matches stems.DRUM_PIECE / render_studio).
_DRUM_PRESET = {36: "analog_kick", 42: "noise_hat", 38: "snare", 41: "tom",
                39: "clap", 45: "tom"}
# Pitched fx one-shots carry their own channel; preset by channel.
_CH_PRESET = {1: "sub_bass", 2: "warm_keys", 3: "supersaw_lead", 4: "glass_pad"}


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
        self._presets: Dict[str, str] = {}     # tag (stem name) -> preset name
        self._cache: Dict[Tuple, np.ndarray] = {}
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

    # -- timbre routing -------------------------------------------------------

    def set_preset(self, tag: str, preset: str) -> None:
        """Route a stem tag to a timbre preset (live timbre_morph follows)."""
        if preset:
            self._presets[tag] = preset

    def _preset_for(self, ev: MusicEvent) -> str:
        if ev.channel == 10:
            return _DRUM_PRESET.get(ev.a, "analog_kick")
        return self._presets.get(ev.tag) or _CH_PRESET.get(ev.channel, "")

    def _render(self, ev: MusicEvent) -> np.ndarray:
        """The note buffer for this event: rich preset when routed (cached on a
        quantised key), the simple shared voice otherwise."""
        name = self._preset_for(ev) if self._presets else ""
        if name:
            try:
                from .timbre import PRESETS, render
                if name in PRESETS:
                    vel_q = (int(ev.b) // 8) * 8 + 4          # 16 buckets
                    key = (name, int(ev.a), round(float(ev.dur), 2), vel_q)
                    buf = self._cache.get(key)
                    if buf is None:
                        buf = render(PRESETS[name], int(ev.a), float(ev.dur),
                                     vel_q).astype(np.float32)
                        if len(self._cache) >= CACHE_MAX:
                            self._cache.pop(next(iter(self._cache)))
                        self._cache[key] = buf
                    return buf
            except Exception:
                pass                              # fall through to simple voices
        return render_note(ev.channel, ev.a, ev.dur, ev.b).astype(np.float32)

    # -- Backend interface --------------------------------------------------

    def send(self, ev: MusicEvent) -> None:
        if not self.ok or ev.kind != "note_on" or not ev.dur:
            return
        sig = self._render(ev)
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
