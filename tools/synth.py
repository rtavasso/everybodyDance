"""Tiny offline synth: render logged MusicEvents (with timestamps) to a WAV.

Voices live in everybody_dance/voices.py (shared with the real-time sandbox) so
the offline preview and the live instrument sound identical.
"""

from __future__ import annotations

import wave
from typing import List

import numpy as np

from everybody_dance.voices import SR, render_note


def render(events: List, out_path: str, tail: float = 1.0) -> float:
    notes = [e for e in events if e.kind in ("note_on", "note_off")]
    if not notes:
        raise ValueError("no notes to render")
    end = max(e.t for e in notes) + tail
    buf = np.zeros(int(end * SR) + SR)

    active = {}   # (ch, pitch) -> (t_on, vel)
    rendered = []
    for e in sorted(notes, key=lambda x: (x.t, x.kind == "note_on")):
        key = (e.channel, e.a)
        if e.kind == "note_on":
            active[key] = (e.t, e.b)
        else:
            if key in active:
                t_on, vel = active.pop(key)
                rendered.append((e.channel, e.a, t_on, e.t - t_on, vel))
    for (ch, pitch), (t_on, vel) in active.items():        # held to the end
        rendered.append((ch, pitch, t_on, end - t_on, vel))

    for ch, pitch, t_on, dur, vel in rendered:
        sig = render_note(ch, pitch, dur, vel)
        i0 = int(t_on * SR)
        i1 = i0 + len(sig)
        if i1 <= len(buf):
            buf[i0:i1] += sig

    peak = np.max(np.abs(buf)) + 1e-9
    pcm = (buf / peak * 0.95 * 32767).astype(np.int16)
    with wave.open(out_path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return end
