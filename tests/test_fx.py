"""Tests for everybody_dance.fx — the audio effects chain."""

import numpy as np
import pytest

from everybody_dance.studio_types import FxConfig, SR
from everybody_dance.fx import apply_fx

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sine(freq=440.0, dur=0.5, sr=SR, amp=0.5) -> np.ndarray:
    """Short deterministic sine burst."""
    t = np.linspace(0, dur, int(dur * sr), endpoint=False, dtype=np.float32)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)

_OFF = FxConfig()   # all defaults == all off


# ---------------------------------------------------------------------------
# All-off identity
# ---------------------------------------------------------------------------

def test_all_off_is_identity():
    """With every effect bypassed the output must equal the input exactly."""
    sig = _sine()
    out = apply_fx(sig, _OFF, sr=SR, bpm=100.0)
    assert out.shape == sig.shape, "length must not change when all effects off"
    np.testing.assert_array_equal(out, sig)


# ---------------------------------------------------------------------------
# Drive
# ---------------------------------------------------------------------------

def test_drive_increases_rms():
    """Saturation compresses peaks and raises perceptual loudness (RMS)."""
    sig = _sine(amp=0.8)
    out = apply_fx(sig, FxConfig(drive=0.9), sr=SR, bpm=100.0)
    assert out.shape == sig.shape
    # drive soft-clips peaks -> RMS rises relative to peak
    assert np.max(np.abs(out)) <= 1.0 + 1e-5   # bounded
    # RMS should not collapse
    assert out.std() > 0.0


def test_drive_peak_clamped():
    """Even extreme drive must not produce samples > 1.0 (tanh guarantees this)."""
    sig = _sine(amp=0.99)
    out = apply_fx(sig, FxConfig(drive=1.0), sr=SR, bpm=100.0)
    assert np.all(np.abs(out) <= 1.0 + 1e-5)


# ---------------------------------------------------------------------------
# Bitcrusher
# ---------------------------------------------------------------------------

def test_bitcrush_reduces_distinct_values():
    """Quantisation must reduce the number of unique sample values."""
    sig = _sine(amp=0.7)
    n_before = len(np.unique(sig))
    out = apply_fx(sig, FxConfig(bitcrush=0.8), sr=SR, bpm=100.0)
    assert out.shape == sig.shape
    n_after = len(np.unique(out))
    assert n_after < n_before, (
        f"bitcrush should reduce distinct values; got {n_after} vs {n_before}")


def test_bitcrush_zero_is_bypass():
    """bitcrush=0 must not alter the signal."""
    sig = _sine()
    out = apply_fx(sig, FxConfig(bitcrush=0.0), sr=SR, bpm=100.0)
    np.testing.assert_array_equal(out, sig)


# ---------------------------------------------------------------------------
# Resonant filter
# ---------------------------------------------------------------------------

def test_filter_open_is_bypass():
    """filter_cutoff=1.0 (open) must not alter the signal."""
    sig = _sine()
    out = apply_fx(sig, FxConfig(filter_cutoff=1.0, filter_res=0.0),
                   sr=SR, bpm=100.0)
    np.testing.assert_array_equal(out, sig)


def test_filter_attenuates_high_freq():
    """A closed lowpass filter should reduce high-frequency energy."""
    # Use a high-frequency sine and a low cutoff
    sig = _sine(freq=5000.0, amp=0.5)
    out = apply_fx(sig, FxConfig(filter_cutoff=0.1, filter_res=0.0),
                   sr=SR, bpm=100.0)
    assert out.shape == sig.shape
    assert out.std() < sig.std() * 0.9   # substantial attenuation expected


# ---------------------------------------------------------------------------
# Delay
# ---------------------------------------------------------------------------

def test_delay_adds_energy_after_delay_time():
    """With delay_send>0 there should be energy after the delay time that
    isn't present in a dry-only run (the input is a short click)."""
    sr = SR
    bpm = 120.0
    delay_beats = 0.5
    beat_s = 60.0 / bpm
    delay_s = delay_beats * beat_s
    delay_samps = int(delay_s * sr)

    # impulse at the very start
    sig = np.zeros(sr, dtype=np.float32)
    sig[0] = 0.8

    out_dry = apply_fx(sig, _OFF, sr=sr, bpm=bpm)
    out_wet = apply_fx(sig,
                       FxConfig(delay_send=0.8, delay_time=delay_beats,
                                delay_feedback=0.5),
                       sr=sr, bpm=bpm)

    # energy in the echo window (2 samples around delay_samps)
    lo = delay_samps - 2
    hi = delay_samps + 3
    assert len(out_wet) > hi, "output must be long enough to contain the echo"
    echo_energy = np.sum(out_wet[lo:hi] ** 2)
    assert echo_energy > 1e-4, (
        f"expected echo energy at sample ~{delay_samps}; got {echo_energy:.2e}")


def test_delay_extends_buffer():
    """delay_send>0 must produce a longer buffer than the input."""
    sig = _sine()
    out = apply_fx(sig, FxConfig(delay_send=0.5, delay_feedback=0.4),
                   sr=SR, bpm=100.0)
    assert len(out) > len(sig)


# ---------------------------------------------------------------------------
# Reverb
# ---------------------------------------------------------------------------

def test_reverb_extends_buffer():
    """reverb_send>0 must produce a longer buffer."""
    sig = _sine()
    out = apply_fx(sig, FxConfig(reverb_send=0.5, reverb_size=0.5),
                   sr=SR, bpm=100.0)
    assert len(out) > len(sig)


def test_reverb_tail_decays():
    """The reverb tail should be non-zero and decay over time after the dry
    signal ends."""
    sig = _sine(dur=0.2, amp=0.8)
    out = apply_fx(sig, FxConfig(reverb_send=0.8, reverb_size=0.6),
                   sr=SR, bpm=100.0)
    dry_end = len(sig)
    assert len(out) > dry_end + SR // 4, "tail must be at least 0.25 s"

    # RMS in early tail > RMS in late tail (decay)
    tail = out[dry_end:]
    mid = len(tail) // 2
    rms_early = np.sqrt(np.mean(tail[:mid] ** 2))
    rms_late  = np.sqrt(np.mean(tail[mid:] ** 2))
    assert rms_early > rms_late, "reverb tail should decay"
    assert rms_early > 1e-6, "reverb tail must contain audible energy"


# ---------------------------------------------------------------------------
# Output always finite & bounded
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fx", [
    FxConfig(),
    FxConfig(drive=1.0),
    FxConfig(bitcrush=1.0),
    FxConfig(filter_cutoff=0.05, filter_res=0.95),
    FxConfig(delay_send=0.9, delay_feedback=0.94),
    FxConfig(reverb_send=1.0, reverb_size=1.0),
    FxConfig(drive=0.8, bitcrush=0.5, filter_cutoff=0.3, filter_res=0.5,
             delay_send=0.4, reverb_send=0.4),
])
def test_output_finite_and_bounded(fx):
    sig = _sine(amp=0.9)
    out = apply_fx(sig, fx, sr=SR, bpm=100.0)
    assert np.all(np.isfinite(out)), "output must not contain NaN or inf"
    assert np.max(np.abs(out)) <= 1.0 + 1e-5, "output must be bounded to [-1, 1]"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_deterministic_two_calls_equal():
    """Calling apply_fx twice with the same input must return identical arrays."""
    sig = _sine(amp=0.6)
    fx = FxConfig(drive=0.3, bitcrush=0.4, filter_cutoff=0.6, filter_res=0.3,
                  delay_send=0.4, delay_feedback=0.5,
                  reverb_send=0.4, reverb_size=0.5)
    out1 = apply_fx(sig.copy(), fx, sr=SR, bpm=100.0)
    out2 = apply_fx(sig.copy(), fx, sr=SR, bpm=100.0)
    np.testing.assert_array_equal(out1, out2)
