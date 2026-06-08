"""Parametric multi-instrument synth engine for the everybodyDance studio.

Implements `render_note` matching the `SynthFn` Protocol in studio_types.
All synthesis is numpy-only; any noise uses a fixed seed so calls are
deterministic. Samples are bounded to roughly |x| < 2.
"""

from __future__ import annotations

import numpy as np

from everybody_dance.studio_types import SR, TimbreConfig

# Public list of supported instrument names.
INSTRUMENTS: list[str] = [
    "sine", "saw", "square", "pluck", "pad",
    "fm", "bass", "bell", "kick", "snare", "hat", "noise",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _midi_hz(n: int) -> float:
    """MIDI note number -> frequency in Hz."""
    return 440.0 * 2.0 ** ((n - 69) / 12.0)


def _t(n: int, sr: int) -> np.ndarray:
    """Time axis for n samples at sr."""
    return np.arange(n, dtype=np.float32) / sr


def _adsr(n: int, attack: float, decay: float, sustain: float,
          release: float, dur_s: float, sr: int) -> np.ndarray:
    """Full ADSR envelope.

    The note sustains until dur_s then releases.  The total buffer length
    n already includes the release tail (dur_s + release) clipped to 4 s.
    """
    env = np.zeros(n, dtype=np.float32)

    a = int(min(attack, dur_s) * sr)
    d = int(min(decay, max(dur_s - attack, 0.0)) * sr)
    hold_end = int(dur_s * sr)          # sample where sustain phase ends
    rel_len = n - hold_end              # remaining samples = release tail

    # Attack
    if a > 0:
        env[:a] = np.linspace(0.0, 1.0, a, dtype=np.float32)
    # Decay
    d_end = a + d
    if d > 0:
        env[a:d_end] = np.linspace(1.0, sustain, d, dtype=np.float32)
    # Sustain
    if hold_end > d_end:
        env[d_end:hold_end] = sustain
    # Release
    if rel_len > 0 and n > hold_end:
        env[hold_end:n] = np.linspace(sustain, 0.0, rel_len, dtype=np.float32)

    return env


def _biquad_lpf(signal: np.ndarray, cutoff_hz: float, q: float,
                sr: int) -> np.ndarray:
    """Simple biquad low-pass filter (Direct-Form I, stable, numpy loop-free
    via explicit sample iteration wrapped in a tight Python loop).

    For CPU efficiency on large buffers we use cascaded vectorised segments
    of 256 samples that re-seed state -- an approximation that is stable and
    cheap without scipy.  For a perfect implementation we iterate sample by
    sample (still fast enough for our buffer sizes ~ 10-100k samples).
    """
    cutoff_hz = float(np.clip(cutoff_hz, 20.0, sr * 0.499))
    q = float(np.clip(q, 0.5, 20.0))

    # Bilinear-transform biquad coefficients
    w0 = 2.0 * np.pi * cutoff_hz / sr
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / (2.0 * q)

    b0 = (1.0 - cos_w0) / 2.0
    b1 = 1.0 - cos_w0
    b2 = (1.0 - cos_w0) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha

    b0 /= a0; b1 /= a0; b2 /= a0
    a1 /= a0; a2 /= a0

    out = np.empty_like(signal)
    x1 = x2 = y1 = y2 = 0.0
    for i, x0 in enumerate(signal):
        y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        out[i] = y0
        x2 = x1; x1 = x0
        y2 = y1; y1 = y0

    return out


def _cutoff_hz(cutoff01: float, sr: int) -> float:
    """Map normalised cutoff 0..1 -> musical Hz range ~80..nyquist."""
    # exponential / musical mapping
    low = 80.0
    high = sr * 0.45
    return float(low * (high / low) ** np.clip(cutoff01, 0.001, 1.0))


def _q_from_resonance(res01: float) -> float:
    """Map 0..1 resonance -> biquad Q (0.5 flat -> ~18 sharp peak)."""
    return 0.5 + float(np.clip(res01, 0.0, 1.0)) * 17.5


# ---------------------------------------------------------------------------
# Per-instrument oscillators (return raw un-enveloped signal)
# ---------------------------------------------------------------------------

def _sine(freq: float, n: int, sr: int) -> np.ndarray:
    t = _t(n, sr)
    return np.sin(2.0 * np.pi * freq * t, dtype=np.float32)


def _saw_additive(freq: float, n: int, sr: int, harmonics: int,
                  detune: float) -> np.ndarray:
    """Bandlimited saw via additive synthesis (avoids Nyquist aliasing)."""
    t = _t(n, sr)
    nyquist = sr / 2.0
    sig = np.zeros(n, dtype=np.float32)
    for k in range(1, harmonics + 1):
        if freq * k >= nyquist:
            break
        amp = 1.0 / k
        # small detune between even/odd harmonics for width
        detune_hz = freq * detune * 0.01 * (1 if k % 2 == 0 else -1)
        sig += amp * np.sin(2.0 * np.pi * (freq * k + detune_hz) * t)
    return (sig * (2.0 / np.pi)).astype(np.float32)


def _square_additive(freq: float, n: int, sr: int,
                     harmonics: int) -> np.ndarray:
    """Bandlimited square via odd harmonics."""
    t = _t(n, sr)
    nyquist = sr / 2.0
    sig = np.zeros(n, dtype=np.float32)
    for k in range(1, harmonics * 2, 2):   # odd only
        if freq * k >= nyquist:
            break
        sig += (1.0 / k) * np.sin(2.0 * np.pi * freq * k * t)
    return (sig * (4.0 / np.pi)).astype(np.float32)


def _fm(freq: float, n: int, sr: int, ratio: float,
        index: float) -> np.ndarray:
    """Classic 2-op FM synthesis."""
    t = _t(n, sr)
    mod = np.sin(2.0 * np.pi * freq * ratio * t)
    return np.sin(2.0 * np.pi * freq * t + index * mod).astype(np.float32)


def _bass_osc(freq: float, n: int, sr: int, harmonics: int) -> np.ndarray:
    """Strong fundamental + modest 2nd harmonic."""
    t = _t(n, sr)
    sig = 0.9 * np.sin(2.0 * np.pi * freq * t)
    if freq * 2 < sr / 2.0:
        sig += 0.25 * np.sin(2.0 * np.pi * freq * 2 * t)
    # additional harmonics from timbre setting
    for k in range(3, min(harmonics + 1, 6)):
        if freq * k >= sr / 2.0:
            break
        sig += (0.1 / k) * np.sin(2.0 * np.pi * freq * k * t)
    return sig.astype(np.float32)


def _bell_osc(freq: float, n: int, sr: int, harmonics: int) -> np.ndarray:
    """Inharmonic FM/additive bell spectrum."""
    t = _t(n, sr)
    # FM carrier with inharmonic modulator
    mod = np.sin(2.0 * np.pi * freq * 2.756 * t)
    sig = 0.5 * np.sin(2.0 * np.pi * freq * t + 3.5 * mod)
    # Add inharmonic partials
    ratios = [1.0, 2.756, 5.404, 8.933, 13.35]
    amps   = [1.0, 0.60,  0.30,  0.15,  0.07 ]
    for i, (r, a) in enumerate(zip(ratios[:harmonics], amps[:harmonics])):
        if freq * r >= sr / 2.0:
            break
        sig += a * np.sin(2.0 * np.pi * freq * r * t)
    return (sig * 0.35).astype(np.float32)


def _pad_osc(freq: float, n: int, sr: int, harmonics: int,
             detune: float) -> np.ndarray:
    """Soft pad: additive sine harmonics with gentle detuning."""
    t = _t(n, sr)
    nyquist = sr / 2.0
    sig = np.zeros(n, dtype=np.float32)
    amps = [1.0, 0.5, 0.25, 0.12, 0.06, 0.03]
    for k in range(1, harmonics + 1):
        if freq * k >= nyquist:
            break
        amp = amps[k - 1] if k <= len(amps) else 0.02 / k
        det = freq * detune * 0.005 * (k - 1)
        sig += amp * np.sin(2.0 * np.pi * (freq * k + det) * t)
    return sig.astype(np.float32)


def _pluck_ks(freq: float, n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Karplus-Strong plucked string."""
    period = max(1, int(sr / freq))
    # Excitation: short burst of noise in the delay line
    buf = rng.uniform(-1.0, 1.0, period).astype(np.float32)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        sample = buf[i % period]
        out[i] = sample
        # Average (low-pass) feedback
        next_idx = (i + 1) % period
        buf[next_idx] = 0.498 * (buf[next_idx] + sample)
    return out


# ---------------------------------------------------------------------------
# Drum synthesis
# ---------------------------------------------------------------------------

def _kick_drum(n: int, sr: int) -> np.ndarray:
    """Kick: pitched sine with fast pitch drop + body thump."""
    t = _t(n, sr)
    # pitch envelope: 150 Hz -> 45 Hz in first 80 ms
    f = 150.0 * np.exp(-t * 35.0) + 45.0
    # integrate frequency to get phase
    phase = 2.0 * np.pi * np.cumsum(f) / sr
    tone  = np.sin(phase)
    # amplitude envelope: fast transient + medium body
    amp   = np.exp(-t * 14.0)
    # small click at attack (high-freq burst)
    click_len = min(int(0.003 * sr), n)
    click = np.zeros(n, dtype=np.float32)
    click[:click_len] = np.linspace(0.3, 0.0, click_len)
    return (tone * amp + click).astype(np.float32)


def _snare_drum(n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Snare: two sine tones + hi-passed noise (the wire rattle)."""
    t = _t(n, sr)
    # Body: two pitched tones
    tone1 = np.sin(2.0 * np.pi * 185.0 * t) * np.exp(-t * 20.0)
    tone2 = np.sin(2.0 * np.pi * 275.0 * t) * np.exp(-t * 25.0)
    # Snare wire noise
    noise = rng.standard_normal(n).astype(np.float32) * np.exp(-t * 18.0)
    # Crude high-pass on noise: first-order difference
    noise_hp = np.diff(noise, prepend=0.0)
    return (0.5 * (tone1 + tone2) + 0.5 * noise_hp).astype(np.float32)


def _hat_drum(n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Hi-hat: differentiated noise with fast exponential decay."""
    noise = rng.standard_normal(n).astype(np.float32)
    noise_hp = np.diff(noise, prepend=0.0)
    t = _t(n, sr)
    return (noise_hp * np.exp(-t * 55.0) * 0.6).astype(np.float32)


def _noise_drum(n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Generic noise hit: white noise with short decay."""
    t = _t(n, sr)
    return (rng.standard_normal(n).astype(np.float32)
            * np.exp(-t * 20.0) * 0.5)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_note(pitch: int, dur_s: float, vel: int,
                timbre: TimbreConfig, sr: int = SR) -> np.ndarray:
    """Render one note to a mono float32 buffer.

    Gain (timbre.gain) and velocity are applied; result is NOT globally
    normalised.  Percussion instruments ignore `pitch`.  Deterministic.
    """
    # --- guard inputs -------------------------------------------------------
    dur_s = float(np.clip(dur_s, 0.03, 4.0))
    vel   = int(np.clip(vel, 1, 127))
    vel01 = vel / 127.0

    # Fixed RNG seed for determinism
    rng = np.random.default_rng(0)

    inst = timbre.instrument.lower()

    # --- percussion: ignore pitch, fixed buffer length ----------------------
    is_drum = inst in ("kick", "snare", "hat", "noise")

    if is_drum:
        # Drums: natural length determined by instrument, not dur_s/release
        if inst == "kick":
            n = max(1, int(min(dur_s + timbre.release, 0.60) * sr))
            raw = _kick_drum(n, sr)
        elif inst == "snare":
            n = max(1, int(min(dur_s + timbre.release, 0.45) * sr))
            raw = _snare_drum(n, sr, rng)
        elif inst == "hat":
            n = max(1, int(min(dur_s + timbre.release, 0.15) * sr))
            raw = _hat_drum(n, sr, rng)
        else:  # noise
            n = max(1, int(min(dur_s + timbre.release, 0.35) * sr))
            raw = _noise_drum(n, sr, rng)

        # Apply velocity and gain, skip ADSR/filter for drums
        out = (raw * vel01 * timbre.gain).astype(np.float32)
        return np.clip(out, -2.0, 2.0)

    # --- pitched instruments ------------------------------------------------
    freq = _midi_hz(pitch)

    # Total buffer: sustain portion + release tail
    n_hold = max(1, int(dur_s * sr))
    n_rel  = max(0, int(timbre.release * sr))
    n      = n_hold + n_rel

    # Oscillator
    if inst == "sine":
        raw = _sine(freq, n, sr)
    elif inst == "saw":
        raw = _saw_additive(freq, n, sr,
                            max(1, timbre.harmonics), timbre.detune)
    elif inst == "square":
        raw = _square_additive(freq, n, sr, max(1, timbre.harmonics))
    elif inst == "fm":
        raw = _fm(freq, n, sr, timbre.fm_ratio, timbre.fm_index)
    elif inst == "bass":
        raw = _bass_osc(freq, n, sr, max(1, timbre.harmonics))
    elif inst == "bell":
        raw = _bell_osc(freq, n, sr, max(1, timbre.harmonics))
    elif inst == "pad":
        raw = _pad_osc(freq, n, sr, max(1, timbre.harmonics), timbre.detune)
    elif inst == "pluck":
        raw = _pluck_ks(freq, n, sr, rng)
    else:
        # Fallback for unknown: pure sine
        raw = _sine(freq, n, sr)

    # ADSR envelope
    env = _adsr(n, timbre.attack, timbre.decay, timbre.sustain,
                timbre.release, dur_s, sr)
    sig = (raw * env).astype(np.float32)

    # Resonant low-pass filter
    cutoff_hz = _cutoff_hz(timbre.cutoff, sr)
    q = _q_from_resonance(timbre.resonance)
    sig = _biquad_lpf(sig, cutoff_hz, q, sr).astype(np.float32)

    # Velocity + gain
    out = (sig * vel01 * timbre.gain).astype(np.float32)

    # Guard against NaN/Inf and clamp amplitude
    if not np.all(np.isfinite(out)):
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(out, -2.0, 2.0)
