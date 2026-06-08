"""Tests for everybody_dance.synth_engine.

Covers: buffer shape/type/finiteness, per-instrument sanity, ADSR tail,
drum pitch-independence, velocity loudness, and determinism.
"""

from __future__ import annotations

import numpy as np
import pytest

from everybody_dance.studio_types import SR, TimbreConfig
from everybody_dance.synth_engine import INSTRUMENTS, render_note


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tc(instrument: str, **kw) -> TimbreConfig:
    return TimbreConfig(instrument=instrument, **kw)


PITCHED = [i for i in INSTRUMENTS if i not in ("kick", "snare", "hat", "noise")]
DRUMS   = ["kick", "snare", "hat", "noise"]


# ---------------------------------------------------------------------------
# 1. Every instrument: finite, bounded, non-empty, float32
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("inst", INSTRUMENTS)
def test_buffer_properties(inst: str):
    tc = _tc(inst)
    buf = render_note(pitch=60, dur_s=0.2, vel=100, timbre=tc, sr=SR)

    assert buf.dtype == np.float32,     f"{inst}: dtype must be float32"
    assert buf.size  > 0,              f"{inst}: buffer must be non-empty"
    assert np.all(np.isfinite(buf)),   f"{inst}: contains NaN or Inf"
    assert float(np.abs(buf).max()) <= 2.0 + 1e-6, f"{inst}: amplitude out of bounds"


# ---------------------------------------------------------------------------
# 2. Approximate expected length for pitched instruments
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("inst", PITCHED)
def test_pitched_buffer_length(inst: str):
    dur_s   = 0.5
    release = 0.2
    tc      = _tc(inst, release=release)
    buf     = render_note(pitch=60, dur_s=dur_s, vel=100, timbre=tc, sr=SR)

    expected = int((dur_s + release) * SR)
    # Allow ±2 samples for rounding
    assert abs(buf.size - expected) <= 2, (
        f"{inst}: expected ~{expected} samples, got {buf.size}"
    )


# ---------------------------------------------------------------------------
# 3. Determinism – two identical calls must produce bit-for-bit equal results
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("inst", INSTRUMENTS)
def test_determinism(inst: str):
    tc  = _tc(inst)
    a   = render_note(pitch=60, dur_s=0.3, vel=90, timbre=tc, sr=SR)
    b   = render_note(pitch=60, dur_s=0.3, vel=90, timbre=tc, sr=SR)
    assert np.array_equal(a, b), f"{inst}: repeated calls differ"


# ---------------------------------------------------------------------------
# 4. ADSR release extends the tail beyond dur_s
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("inst", PITCHED)
def test_adsr_release_tail(inst: str):
    dur_s   = 0.3
    release = 0.25
    tc_long = _tc(inst, release=release)
    tc_none = _tc(inst, release=0.0)

    buf_long = render_note(60, dur_s, 100, tc_long, SR)
    buf_none = render_note(60, dur_s, 100, tc_none, SR)

    # With release > 0 the buffer must be strictly longer
    assert buf_long.size > buf_none.size, (
        f"{inst}: release tail did not extend buffer"
    )


# ---------------------------------------------------------------------------
# 5. Drums ignore pitch (kick at 40 == kick at 60, etc.)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("inst", DRUMS)
def test_drums_ignore_pitch(inst: str):
    tc = _tc(inst)
    a  = render_note(pitch=40, dur_s=0.15, vel=100, timbre=tc, sr=SR)
    b  = render_note(pitch=60, dur_s=0.15, vel=100, timbre=tc, sr=SR)
    assert np.array_equal(a, b), f"{inst}: drum output differs with pitch"


# ---------------------------------------------------------------------------
# 6. Higher velocity -> louder (strictly greater RMS)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("inst", INSTRUMENTS)
def test_velocity_scales_loudness(inst: str):
    tc   = _tc(inst)
    lo   = render_note(60, 0.2, 30,  tc, SR)
    hi   = render_note(60, 0.2, 120, tc, SR)

    rms_lo = float(np.sqrt(np.mean(lo ** 2)))
    rms_hi = float(np.sqrt(np.mean(hi ** 2)))

    assert rms_hi > rms_lo, (
        f"{inst}: higher velocity did not produce louder output "
        f"(lo rms={rms_lo:.6f}, hi rms={rms_hi:.6f})"
    )


# ---------------------------------------------------------------------------
# 7. Short duration (0.03 s) does not crash or produce invalid output
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("inst", INSTRUMENTS)
def test_short_duration(inst: str):
    tc  = _tc(inst, attack=0.002, decay=0.01, release=0.01)
    buf = render_note(60, 0.03, 80, tc, SR)
    assert buf.size > 0
    assert np.all(np.isfinite(buf))


# ---------------------------------------------------------------------------
# 8. Long duration (4.0 s) does not crash
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("inst", INSTRUMENTS)
def test_long_duration(inst: str):
    tc  = _tc(inst, sustain=0.7, release=0.3)
    buf = render_note(60, 4.0, 100, tc, SR)
    assert buf.size > 0
    assert np.all(np.isfinite(buf))


# ---------------------------------------------------------------------------
# 9. INSTRUMENTS list completeness
# ---------------------------------------------------------------------------

def test_instruments_list():
    required = {"sine", "saw", "square", "pluck", "pad", "fm",
                "bass", "bell", "kick", "snare", "hat", "noise"}
    assert required.issubset(set(INSTRUMENTS)), (
        f"Missing instruments: {required - set(INSTRUMENTS)}"
    )


# ---------------------------------------------------------------------------
# 10. FM parameters are honoured (different fm_ratio gives different output)
# ---------------------------------------------------------------------------

def test_fm_ratio_changes_sound():
    a = render_note(60, 0.3, 100, _tc("fm", fm_ratio=2.0, fm_index=3.0), SR)
    b = render_note(60, 0.3, 100, _tc("fm", fm_ratio=3.5, fm_index=3.0), SR)
    assert not np.array_equal(a, b), "fm: different fm_ratio gave identical output"


# ---------------------------------------------------------------------------
# 11. SynthFn protocol conformance (callable signature check)
# ---------------------------------------------------------------------------

def test_synthfn_protocol():
    from everybody_dance.studio_types import SynthFn
    # render_note is duck-compatible – call it through the Protocol type hint
    fn: SynthFn = render_note   # type: ignore[assignment]
    buf = fn(pitch=64, dur_s=0.1, vel=90, timbre=TimbreConfig(), sr=SR)
    assert buf.size > 0
