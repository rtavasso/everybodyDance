"""Customizable timbre + global FX (numpy-only, deterministic).

Generalises the fixed voices in voices.py into a small subtractive/additive
synth (:class:`TimbrePreset` + :func:`render`) plus a send-style effects bus
(:class:`FXRack`). voices.py stays the simple back-compat path; this is the
palette the studio reaches for when it wants character. Any randomness is
seeded so renders are byte-reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .voices import SR, midi_hz

_SEED = 1234567  # fixed: noise must be byte-reproducible


# -- oscillators -----------------------------------------------------------

def _osc(shape: str, phase: np.ndarray) -> np.ndarray:
    """One oscillator over a 0..1 cyclic phase (cheap, not antialiased)."""
    if shape == "sine":
        return np.sin(2 * np.pi * phase)
    if shape == "saw":
        return 2.0 * (phase - np.floor(phase + 0.5))
    if shape == "square":
        return np.where((phase % 1.0) < 0.5, 1.0, -1.0)
    if shape == "triangle":
        return 2.0 * np.abs(2.0 * (phase - np.floor(phase + 0.5))) - 1.0
    if shape == "noise":
        return np.random.default_rng(_SEED).normal(0, 1, phase.shape)
    raise ValueError(f"unknown wave shape: {shape}")


def _adsr(n: int, a: float, d: float, s: float, r: float) -> np.ndarray:
    """ADSR envelope of length n (a/d/r seconds, s sustain 0..1)."""
    e = np.full(n, s, dtype=np.float64)
    ai = min(int(a * SR), n)
    di = min(int(d * SR), max(n - ai, 0))
    ri = min(int(r * SR), n)
    if ai > 0:
        e[:ai] = np.linspace(0, 1, ai)
    if di > 0:
        e[ai:ai + di] = np.linspace(1, s, di)
    if ri > 0:
        e[-ri:] *= np.linspace(1, 0, ri)
    return e


def _lp1(x: np.ndarray, cutoff: np.ndarray, reso: float) -> np.ndarray:
    """Resonant-ish one-pole lowpass; cutoff (Hz) may be per-sample.

    Resonance feeds a little of the previous output back to peak near cutoff.
    Stable for reso in 0..1.
    """
    fc = np.clip(cutoff, 10.0, SR * 0.45)
    alpha = 1.0 - np.exp(-2 * np.pi * fc / SR)        # per-sample coefficient
    y = np.empty_like(x)
    prev = 0.0
    fb = 0.9 * reso
    for i in range(x.shape[0]):
        prev += alpha[i] * (x[i] - prev + fb * (prev - x[i]))
        y[i] = prev
    return y


# -- drums -----------------------------------------------------------------

def _kick(dur, vel, freq):
    n = max(int(min(dur, 0.35) * SR), 1)
    t = np.arange(n) / SR
    f = freq * np.exp(-t * 30) + 45
    body = np.sin(2 * np.pi * f * t) * np.exp(-t * 14)
    click = np.exp(-t * 400) * 0.3
    return (body + click) * (vel / 127.0)


def _tom(dur, vel, freq):
    n = max(int(min(dur, 0.3) * SR), 1)
    t = np.arange(n) / SR
    f = freq * np.exp(-t * 12) + 80
    return np.sin(2 * np.pi * f * t) * np.exp(-t * 9) * (vel / 127.0)


def _hat(dur, vel):
    n = max(int(min(dur, 0.12) * SR), 1)
    noise = np.diff(np.random.default_rng(_SEED).normal(0, 1, n), prepend=0)
    return noise * np.exp(-np.arange(n) / SR * 60) * (vel / 127.0) * 0.5


def _snare(dur, vel):
    n = max(int(min(dur, 0.25) * SR), 1)
    t = np.arange(n) / SR
    noise = np.random.default_rng(_SEED).normal(0, 1, n) * np.exp(-t * 22)
    tone = np.sin(2 * np.pi * 185 * t) * np.exp(-t * 30) * 0.6
    return (noise + tone) * (vel / 127.0) * 0.7


def _clap(dur, vel):
    n = max(int(min(dur, 0.3) * SR), 1)
    t = np.arange(n) / SR
    rng = np.random.default_rng(_SEED)
    burst = np.zeros(n)
    for off in (0.0, 0.01, 0.02, 0.03):                # stacked short bursts
        env = np.exp(-np.clip(t - off, 0, None) * 90) * (t >= off)
        burst += rng.normal(0, 1, n) * env
    tail = np.random.default_rng(_SEED + 1).normal(0, 1, n) * np.exp(-t * 16)
    return (burst + 0.5 * tail) * (vel / 127.0) * 0.4


def _rim(dur, vel):
    n = max(int(min(dur, 0.08) * SR), 1)
    t = np.arange(n) / SR
    tone = np.sin(2 * np.pi * 1700 * t) + 0.7 * np.sin(2 * np.pi * 520 * t)
    return tone * np.exp(-t * 120) * (vel / 127.0) * 0.6


_DRUMS = {"kick": _kick, "snare": _snare, "hat": _hat,
          "tom": _tom, "clap": _clap, "rim": _rim}


# -- preset ----------------------------------------------------------------

@dataclass
class TimbrePreset:
    name: str
    waves: list = field(default_factory=lambda: [("sine", 1.0)])
    detune_cents: float = 0.0
    voices: int = 1
    attack: float = 0.005
    decay: float = 0.1
    sustain: float = 0.7
    release: float = 0.08
    cutoff: float = 4000.0
    resonance: float = 0.0
    cutoff_env: float = 0.0
    drive: float = 0.0
    noise: float = 0.0
    gain: float = 1.0
    percussive: bool = False
    drum_kind: str = ""
    reverb_send: float = 0.0
    delay_send: float = 0.0


def render(preset: TimbrePreset, pitch: int, dur: float, vel: float, *,
           cutoff_mul: float = 1.0, drive_mul: float = 1.0,
           detune_mul: float = 1.0) -> np.ndarray:
    """One note's mono float32 waveform (gain applied, not normalised).

    The *_mul kwargs let per-note movement automation nudge timbre (e.g. energy
    -> brighter cutoff). Percussive presets dispatch by drum_kind; pitch can
    select a tom/kick fundamental.
    """
    vel = float(vel)
    if preset.percussive:
        kind = preset.drum_kind or "kick"
        if kind in ("kick", "tom"):
            base = 120.0 if kind == "kick" else 180.0
            freq = base * (midi_hz(max(pitch, 24)) / midi_hz(45)) if pitch else base
            sig = _DRUMS[kind](max(dur, 0.05), vel, freq)
        else:
            sig = _DRUMS[kind](max(dur, 0.05), vel)
        return np.clip(sig * preset.gain, -1.5, 1.5).astype(np.float32)

    dur = max(dur, 0.03)
    n = max(int(dur * SR), 1)
    rel = int(preset.release * SR)
    n_tot = n + rel
    t = np.arange(n_tot) / SR
    freq = midi_hz(pitch)

    # oscillator mix with detuned unison voices
    sig = np.zeros(n_tot)
    nv = max(int(preset.voices), 1)
    spread = preset.detune_cents * detune_mul
    for vi in range(nv):
        if nv > 1:
            c = spread * (vi / (nv - 1) - 0.5)         # -spread/2 .. +spread/2
        else:
            c = 0.0
        f = freq * 2 ** (c / 1200.0)
        phase = f * t
        for shape, amp in preset.waves:
            sig += (amp / nv) * _osc(shape, phase if shape != "noise" else t)

    if preset.noise > 0:
        sig += preset.noise * np.random.default_rng(_SEED).normal(0, 1, n_tot)

    env = _adsr(n_tot, preset.attack, preset.decay, preset.sustain,
                preset.release)

    # envelope opens the filter: cutoff_env scales cutoff up with the envelope
    cut = preset.cutoff * cutoff_mul * (1.0 + preset.cutoff_env * env)
    sig = _lp1(sig, cut, preset.resonance)

    drive = preset.drive * drive_mul
    if drive > 0:
        sig = np.tanh(sig * (1.0 + drive)) / np.tanh(1.0 + drive)

    sig *= env * (vel / 127.0) * preset.gain
    return np.clip(sig, -1.5, 1.5).astype(np.float32)


def render_event(preset: TimbrePreset, ev, **mods) -> np.ndarray:
    """Render a note_on MusicEvent (uses ev.a pitch, ev.b vel, ev.dur)."""
    return render(preset, ev.a, ev.dur or 0.25, ev.b, **mods)


# -- global FX -------------------------------------------------------------

class FXRack:
    """Send-style global effects on a mixed mono buffer."""

    def __init__(self, sr: int = SR):
        self.sr = sr

    def _delay(self, buf, wet, time_s, feedback):
        d = max(int(time_s * self.sr), 1)
        out = buf.astype(np.float64).copy()
        fb = np.clip(feedback, 0.0, 0.95)
        # feedback comb: each pass adds a decayed, delayed copy
        tap = out.copy()
        for _ in range(6):
            tap = np.concatenate([np.zeros(d), tap[:-d]]) * fb
            if np.max(np.abs(tap)) < 1e-6:
                break
            out += tap
        return buf + wet * (out - buf)

    def _comb(self, buf, d, g):
        out = buf.astype(np.float64).copy()
        for i in range(d, out.shape[0]):
            out[i] += g * out[i - d]
        return out

    def _allpass(self, buf, d, g):
        out = np.zeros_like(buf, dtype=np.float64)
        x = buf.astype(np.float64)
        for i in range(out.shape[0]):
            xd = x[i - d] if i >= d else 0.0
            yd = out[i - d] if i >= d else 0.0
            out[i] = -g * x[i] + xd + g * yd
        return out

    def _reverb(self, buf, wet):
        # Schroeder: parallel feedback combs -> series allpasses
        combs = [(1116, 0.84), (1188, 0.82), (1277, 0.80), (1356, 0.78)]
        acc = np.zeros_like(buf, dtype=np.float64)
        for d, g in combs:
            acc += self._comb(buf, max(d * self.sr // 44100, 1), g)
        acc /= len(combs)
        for d, g in [(225, 0.5), (556, 0.5)]:
            acc = self._allpass(acc, max(d * self.sr // 44100, 1), g)
        acc *= 0.25
        return buf + wet * acc

    def process(self, buf: np.ndarray, *, reverb: float = 0.0, delay: float = 0.0,
                delay_time_s: float = 0.33, feedback: float = 0.35,
                drive: float = 0.0, bitcrush: float = 0.0) -> np.ndarray:
        """Apply global FX; each param 0..1 scales its wet amount.

        All zero -> buffer returned ~unchanged. Deterministic and bounded.
        """
        x = np.asarray(buf, dtype=np.float64).copy()
        if delay > 0:
            x = self._delay(x, np.clip(delay, 0, 1), delay_time_s, feedback)
        if reverb > 0:
            x = self._reverb(x, np.clip(reverb, 0, 1))
        if drive > 0:
            d = 1.0 + 8.0 * np.clip(drive, 0, 1)
            x = np.tanh(x * d) / np.tanh(d)
        if bitcrush > 0:
            bc = np.clip(bitcrush, 0, 1)
            bits = 16 - int(round(13 * bc))            # 16 -> 3 bits
            step = 2.0 ** (1 - bits)
            crushed = np.round(x / step) * step
            ds = 1 + int(round(15 * bc))               # decimation factor
            if ds > 1:
                idx = (np.arange(x.shape[0]) // ds) * ds
                crushed = crushed[idx]
            x = x + bc * (crushed - x)
        return np.clip(x, -1.5, 1.5).astype(np.float32)


# -- palette ---------------------------------------------------------------

PRESETS: dict = {
    "sub_bass": TimbrePreset(
        "sub_bass", waves=[("sine", 1.0), ("triangle", 0.15)],
        attack=0.004, decay=0.08, sustain=0.85, release=0.06,
        cutoff=600, cutoff_env=0.5, drive=0.2, gain=0.95),
    "reese_bass": TimbrePreset(
        "reese_bass", waves=[("saw", 0.6), ("saw", 0.6)], voices=3,
        detune_cents=22, attack=0.006, decay=0.15, sustain=0.8, release=0.08,
        cutoff=900, resonance=0.35, cutoff_env=1.2, drive=0.6, gain=0.7,
        reverb_send=0.1),
    "warm_keys": TimbrePreset(
        "warm_keys", waves=[("triangle", 0.7), ("sine", 0.4), ("saw", 0.12)],
        attack=0.01, decay=0.25, sustain=0.55, release=0.25,
        cutoff=3200, resonance=0.15, cutoff_env=0.6, drive=0.1, gain=0.55,
        reverb_send=0.25, delay_send=0.1),
    "glass_pad": TimbrePreset(
        "glass_pad", waves=[("sine", 0.5), ("triangle", 0.5), ("saw", 0.1)],
        voices=2, detune_cents=10, attack=0.4, decay=0.5, sustain=0.7,
        release=0.7, cutoff=2600, resonance=0.2, cutoff_env=0.8, noise=0.015,
        gain=0.4, reverb_send=0.5, delay_send=0.2),
    "supersaw_lead": TimbrePreset(
        "supersaw_lead", waves=[("saw", 0.8)], voices=7, detune_cents=28,
        attack=0.008, decay=0.18, sustain=0.6, release=0.12,
        cutoff=5000, resonance=0.25, cutoff_env=1.0, drive=0.3, gain=0.45,
        reverb_send=0.2, delay_send=0.25),
    "pluck_lead": TimbrePreset(
        "pluck_lead", waves=[("square", 0.5), ("saw", 0.4)],
        attack=0.002, decay=0.12, sustain=0.0, release=0.06,
        cutoff=4200, resonance=0.4, cutoff_env=1.5, drive=0.25, gain=0.6,
        delay_send=0.3),
    "analog_kick": TimbrePreset(
        "analog_kick", percussive=True, drum_kind="kick", gain=0.95),
    "noise_hat": TimbrePreset(
        "noise_hat", percussive=True, drum_kind="hat", gain=0.9),
    "snare": TimbrePreset(
        "snare", percussive=True, drum_kind="snare", gain=0.9, reverb_send=0.15),
    "tom": TimbrePreset(
        "tom", percussive=True, drum_kind="tom", gain=0.9),
    "clap": TimbrePreset(
        "clap", percussive=True, drum_kind="clap", gain=0.95, reverb_send=0.2),
}
