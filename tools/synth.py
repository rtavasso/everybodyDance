"""Tiny offline synth: render logged MusicEvents (with timestamps) to a WAV.

Timbres here are throwaway -- the real instrument drives external synths. This
exists so musicality can be *heard* (and shipped to the user) during iteration,
not just inferred from stats.
"""

from __future__ import annotations

import wave
from typing import List

import numpy as np

SR = 22050


def _midi_hz(n: int) -> float:
    return 440.0 * 2 ** ((n - 69) / 12.0)


def _env(n: int, attack: float, release: float) -> np.ndarray:
    e = np.ones(n)
    a = min(int(attack * SR), n)
    r = min(int(release * SR), n)
    if a > 0:
        e[:a] = np.linspace(0, 1, a)
    if r > 0:
        e[-r:] *= np.linspace(1, 0, r)
    return e


def _pitched(freq: float, dur: float, vel: float, harmonics, attack, release):
    n = max(int(dur * SR), 1)
    t = np.arange(n) / SR
    sig = np.zeros(n)
    for k, amp in harmonics:
        sig += amp * np.sin(2 * np.pi * freq * k * t)
    sig *= _env(n, attack, release) * (vel / 127.0)
    return sig


def _kick(dur, vel):
    n = max(int(min(dur, 0.3) * SR), 1)
    t = np.arange(n) / SR
    f = 120 * np.exp(-t * 30) + 45
    sig = np.sin(2 * np.pi * f * t) * np.exp(-t * 14) * (vel / 127.0)
    return sig


def _hat(dur, vel):
    n = max(int(min(dur, 0.12) * SR), 1)
    noise = np.random.default_rng(0).normal(0, 1, n)
    noise = np.diff(noise, prepend=0)  # crude high-pass
    return noise * np.exp(-np.arange(n) / SR * 60) * (vel / 127.0) * 0.5


def _tom(dur, vel):
    n = max(int(min(dur, 0.25) * SR), 1)
    t = np.arange(n) / SR
    f = 180 * np.exp(-t * 12) + 80
    return np.sin(2 * np.pi * f * t) * np.exp(-t * 9) * (vel / 127.0)


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
    # held notes never released (e.g. final pad): play to the end
    for (ch, pitch), (t_on, vel) in active.items():
        rendered.append((ch, pitch, t_on, end - t_on, vel))

    for ch, pitch, t_on, dur, vel in rendered:
        dur = max(dur, 0.03)
        if ch == 10:
            if pitch == 36:
                sig = _kick(dur, vel)
            elif pitch == 42:
                sig = _hat(dur, vel)
            else:
                sig = _tom(dur, vel)
            gain = 0.9
        else:
            freq = _midi_hz(pitch)
            if ch == 1:      # bass
                sig = _pitched(freq, dur, vel, [(1, 0.9), (2, 0.25)], 0.005, 0.05)
                gain = 0.8
            elif ch == 2:    # pad / chord
                sig = _pitched(freq, dur, vel, [(1, 0.6), (2, 0.2), (3, 0.1)],
                               0.08, 0.2)
                gain = 0.35
            else:            # lead
                sig = _pitched(freq, dur, vel,
                               [(k, 0.7 / k) for k in range(1, 6)], 0.005, 0.04)
                gain = 0.5
        i0 = int(t_on * SR)
        i1 = i0 + len(sig)
        if i1 <= len(buf):
            buf[i0:i1] += gain * sig

    # normalise
    peak = np.max(np.abs(buf)) + 1e-9
    buf = buf / peak * 0.95
    pcm = (buf * 32767).astype(np.int16)
    with wave.open(out_path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return end
