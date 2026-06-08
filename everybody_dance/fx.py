"""Audio FX chain for the everybodyDance studio.

Serial order: drive -> bitcrush -> resonant filter -> delay -> reverb -> soft-clip.
Each stage is a numpy-only implementation; no scipy.  All processing is
deterministic (fixed-topology, no RNG).  When every parameter is at its default
"off" value the output is bit-for-bit identical to the input (within float32
epsilon and the same length).
"""

from __future__ import annotations

import numpy as np

from everybody_dance.studio_types import FxConfig, SR

# ---------------------------------------------------------------------------
# Drive (soft saturation)
# ---------------------------------------------------------------------------

def _drive(buf: np.ndarray, amount: float) -> np.ndarray:
    """tanh soft-clip with pre-gain proportional to `amount`.

    amount=0 -> bypass (pre-gain=1, but tanh(x)≈x for small x so no change).
    We bias the pre-gain so that at amount=0 the signal passes unmodified.
    """
    if amount <= 0.0:
        return buf
    # pre-gain 1..8 on a curve; post-divide restores rough loudness
    pre = 1.0 + amount * 7.0
    return np.tanh(buf * pre) / np.tanh(np.float32(pre))


# ---------------------------------------------------------------------------
# Bitcrusher (sample-rate decimation + bit-depth reduction)
# ---------------------------------------------------------------------------

def _bitcrush(buf: np.ndarray, amount: float, sr: int) -> np.ndarray:
    """Decimate sample-rate then quantise to fewer bits.

    amount=0 -> bypass.
    amount=1 -> downsample to ~1/8 rate + 3-bit depth.
    """
    if amount <= 0.0:
        return buf
    # downsampling factor 1..8 (integer steps)
    factor = max(1, int(1 + amount * 7))
    # bit depth 16..3
    bits = max(3, int(16 - amount * 13))
    # hold (zero-order) downsampling
    out = buf.copy()
    out[::factor] = buf[::factor]           # keep samples at step
    for k in range(1, factor):
        idx = np.arange(k, len(buf), factor)
        if len(idx):
            out[idx] = out[np.maximum(idx - k, 0)]
    # bit-depth quantisation in [-1, 1)
    levels = 2.0 ** (bits - 1)
    out = np.floor(out * levels + 0.5) / levels
    return out


# ---------------------------------------------------------------------------
# Resonant low-pass filter (state-variable, Chamberlin topology)
# ---------------------------------------------------------------------------

def _svf_lowpass(buf: np.ndarray, cutoff01: float, res: float) -> np.ndarray:
    """Chamberlin state-variable filter – stable up to cutoff01=1.

    cutoff01=1.0, res=0.0 -> bypass (output equals input to float precision).
    cutoff01 maps 0..1 -> ~20 Hz .. Nyquist/2 on a log scale.
    res 0..1 -> Q 0.5..8 (clamped for stability).
    """
    if cutoff01 >= 1.0:
        return buf

    sr_f = float(SR)
    # frequency in Hz: log scale 20 Hz to sr/2
    f_hz = 20.0 * (sr_f / 2.0 / 20.0) ** float(cutoff01)
    f_hz = min(f_hz, sr_f * 0.49)
    # Chamberlin coefficient (small-angle approx; stable for f < sr/6)
    f_coef = 2.0 * np.sin(np.pi * f_hz / sr_f)
    # Q from res; keep >= 0.5 for stability
    q_inv = 1.0 / max(0.5, 0.5 + res * 7.5)

    low, band = 0.0, 0.0
    out = np.empty_like(buf)
    for i, x in enumerate(buf):
        low = low + f_coef * band
        high = x - low - q_inv * band
        band = f_coef * high + band
        out[i] = low
    return out


# ---------------------------------------------------------------------------
# Delay (beat-synced feedback delay with tail extension)
# ---------------------------------------------------------------------------

_MAX_DELAY_TAIL_S = 4.0   # seconds of tail to append (at most)

def _delay(buf: np.ndarray, fx: FxConfig, sr: int, bpm: float) -> np.ndarray:
    """Feedback comb delay.  Buffer is extended to hold the decaying tail."""
    if fx.delay_send <= 0.0:
        return buf

    beat_s = 60.0 / max(bpm, 1.0)
    delay_s = fx.delay_time * beat_s
    delay_samps = max(1, int(round(delay_s * sr)))
    feedback = min(fx.delay_feedback, 0.94)   # stability cap

    # how long does the tail need to be? solve feedback^n < -60 dB
    if feedback > 0.0:
        tail_s = min(-3.0 / np.log10(feedback) * delay_s, _MAX_DELAY_TAIL_S)
    else:
        tail_s = delay_s
    tail_samps = int(tail_s * sr)

    n_out = len(buf) + tail_samps
    wet = np.zeros(n_out, dtype=np.float32)
    dry = np.zeros(n_out, dtype=np.float32)
    dry[:len(buf)] = buf

    line = np.zeros(delay_samps, dtype=np.float32)
    ptr = 0
    for i in range(n_out):
        x_in = dry[i]
        delayed = line[ptr]
        new_sample = x_in + feedback * delayed
        line[ptr] = new_sample
        wet[i] = delayed
        ptr = (ptr + 1) % delay_samps

    return dry + fx.delay_send * wet


# ---------------------------------------------------------------------------
# Reverb (Schroeder: parallel combs + series allpasses)
# ---------------------------------------------------------------------------

# Prime-spaced comb lengths for the reverb (samples at SR=22050, scale later)
_COMB_DELAYS_BASE = [1557, 1617, 1491, 1422]   # Moorer/Schroeder canonical
_AP_DELAYS_BASE   = [225, 556]                  # two allpass stages

_MAX_REVERB_TAIL_S = 4.0

def _reverb(buf: np.ndarray, fx: FxConfig, sr: int) -> np.ndarray:
    """Small Schroeder reverb: 4 parallel comb filters -> 2 series allpasses.

    reverb_size (0..1) scales the delay line lengths.
    reverb_send (0..1) mixes wet signal.
    reverb_send=0 -> bypass (returns buf unchanged).
    """
    if fx.reverb_send <= 0.0:
        return buf

    size = max(0.1, fx.reverb_size)
    sr_scale = sr / 22050.0

    comb_delays = [max(1, int(d * size * sr_scale)) for d in _COMB_DELAYS_BASE]
    ap_delays   = [max(1, int(d * size * sr_scale)) for d in _AP_DELAYS_BASE]

    # RT60-ish feedback: tuned so at size=0.5 -> ~0.8 s decay
    comb_fb = 0.84 + size * 0.12   # 0.84..0.96

    # tail length: -60 dB time for comb filters
    max_delay = max(comb_delays)
    tail_s = min(-3.0 / np.log10(max(comb_fb, 1e-6)) *
                 max_delay / float(sr), _MAX_REVERB_TAIL_S)
    tail_samps = int(tail_s * sr)

    n_out = len(buf) + tail_samps
    in_pad = np.zeros(n_out, dtype=np.float32)
    in_pad[:len(buf)] = buf

    # --- parallel comb filters ---
    comb_sum = np.zeros(n_out, dtype=np.float32)
    for delay in comb_delays:
        line = np.zeros(delay, dtype=np.float32)
        ptr = 0
        tmp = np.empty(n_out, dtype=np.float32)
        for i in range(n_out):
            delayed = line[ptr]
            line[ptr] = in_pad[i] + comb_fb * delayed
            tmp[i] = delayed
            ptr = (ptr + 1) % delay
        comb_sum += tmp
    comb_sum *= 0.25   # average over 4 combs

    # --- series allpass filters (ap_gain=0.5 is classic Schroeder) ---
    ap_gain = 0.5
    sig = comb_sum
    for delay in ap_delays:
        line = np.zeros(delay, dtype=np.float32)
        ptr = 0
        tmp = np.empty(n_out, dtype=np.float32)
        for i in range(n_out):
            delayed = line[ptr]
            v = sig[i] - ap_gain * delayed
            line[ptr] = v
            tmp[i] = delayed + ap_gain * v
            ptr = (ptr + 1) % delay
        sig = tmp

    dry = np.zeros(n_out, dtype=np.float32)
    dry[:len(buf)] = buf
    return dry + fx.reverb_send * sig


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def apply_fx(buf: np.ndarray, fx: FxConfig,
             sr: int = SR, bpm: float = 100.0) -> np.ndarray:
    """Apply the full FX chain to a mono float32 buffer.

    Returns a (potentially longer) float32 buffer.  Deterministic and NaN-free.
    When all params are at off defaults the output equals the input exactly.
    """
    buf = np.asarray(buf, dtype=np.float32)

    # 1. Drive
    buf = _drive(buf, fx.drive)

    # 2. Bitcrusher
    buf = _bitcrush(buf, fx.bitcrush, sr)

    # 3. Resonant low-pass filter
    buf = _svf_lowpass(buf, fx.filter_cutoff, fx.filter_res)

    # 4. Delay (may lengthen)
    buf = _delay(buf, fx, sr, bpm)

    # 5. Reverb (may lengthen further)
    buf = _reverb(buf, fx, sr)

    # 6. Final soft-clip to prevent runaway; skip when no effects touched the
    #    buffer so the all-off identity contract holds exactly.
    any_active = (fx.drive > 0 or fx.bitcrush > 0 or fx.filter_cutoff < 1.0
                  or fx.delay_send > 0 or fx.reverb_send > 0)
    if any_active:
        buf = np.tanh(buf).astype(np.float32)

    return buf
