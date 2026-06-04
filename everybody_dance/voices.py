"""Shared synth voices.

Single source of truth for how a MusicEvent *sounds*, used by both the offline
WAV renderer (tools/synth.py) and the real-time sandbox synth (rtaudio.py).
Timbres are deliberately simple -- the real instrument drives external synths;
these exist so the music can be heard while iterating.
"""

from __future__ import annotations

import numpy as np

SR = 22050


def midi_hz(n: int) -> float:
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


def _pitched(freq, dur, vel, harmonics, attack, release):
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
    return np.sin(2 * np.pi * f * t) * np.exp(-t * 14) * (vel / 127.0)


def _hat(dur, vel):
    n = max(int(min(dur, 0.12) * SR), 1)
    noise = np.diff(np.random.default_rng(0).normal(0, 1, n), prepend=0)
    return noise * np.exp(-np.arange(n) / SR * 60) * (vel / 127.0) * 0.5


def _tom(dur, vel):
    n = max(int(min(dur, 0.25) * SR), 1)
    t = np.arange(n) / SR
    f = 180 * np.exp(-t * 12) + 80
    return np.sin(2 * np.pi * f * t) * np.exp(-t * 9) * (vel / 127.0)


def render_note(channel: int, pitch: int, dur: float, vel: float) -> np.ndarray:
    """One note's waveform (gain applied, not normalised)."""
    dur = max(dur, 0.03)
    if channel == 10:
        if pitch == 36:
            return 0.9 * _kick(dur, vel)
        if pitch == 42:
            return 0.9 * _hat(dur, vel)
        return 0.9 * _tom(dur, vel)
    freq = midi_hz(pitch)
    if channel == 1:        # bass
        return 0.8 * _pitched(freq, dur, vel, [(1, 0.9), (2, 0.25)], 0.005, 0.05)
    if channel == 2:        # pad / chord
        return 0.35 * _pitched(freq, dur, vel, [(1, 0.6), (2, 0.2), (3, 0.1)], 0.08, 0.2)
    return 0.5 * _pitched(freq, dur, vel, [(k, 0.7 / k) for k in range(1, 6)], 0.005, 0.04)
