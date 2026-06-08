"""Tests for the customizable timbre + FX module (deterministic, numpy-only)."""

import numpy as np

from everybody_dance.output import MusicEvent
from everybody_dance.timbre import (
    PRESETS, FXRack, TimbrePreset, render, render_event,
)
from everybody_dance.voices import SR


def _finite_bounded(a, lim=1.5):
    return a.size > 0 and np.all(np.isfinite(a)) and np.abs(a).max() <= lim


def test_every_preset_renders_finite_bounded_nonempty():
    for name, preset in PRESETS.items():
        a = render(preset, 48, 0.4, 100)
        assert a.dtype == np.float32, name
        assert _finite_bounded(a), name


def test_render_is_deterministic():
    p = PRESETS["supersaw_lead"]
    a = render(p, 60, 0.3, 100)
    b = render(p, 60, 0.3, 100)
    assert np.array_equal(a, b)                       # no per-call RNG drift
    # noisy / drum presets too
    for name in ("glass_pad", "noise_hat", "clap", "snare"):
        x = render(PRESETS[name], 50, 0.2, 90)
        y = render(PRESETS[name], 50, 0.2, 90)
        assert np.array_equal(x, y), name


def test_pitched_length_and_adsr_sanity():
    p = PRESETS["warm_keys"]
    dur = 0.5
    a = render(p, 60, dur, 100)
    n = int(dur * SR)
    rel = int(p.release * SR)
    assert n <= a.size <= n + rel + 2                 # body + release tail
    # attack rises from ~0; release tail decays toward ~0
    assert abs(a[0]) < 0.05
    assert np.abs(a[-1]) < np.abs(a[n // 2]) + 1e-6


def test_cutoff_and_drive_mul_change_output():
    p = PRESETS["reese_bass"]
    base = render(p, 40, 0.3, 100)
    bright = render(p, 40, 0.3, 100, cutoff_mul=3.0)
    dirty = render(p, 40, 0.3, 100, drive_mul=4.0)
    detuned = render(p, 40, 0.3, 100, detune_mul=4.0)
    assert not np.array_equal(base, bright)
    assert not np.array_equal(base, dirty)
    assert not np.array_equal(base, detuned)
    # brighter cutoff lets through more high-frequency energy
    assert np.mean(np.abs(np.diff(bright))) > np.mean(np.abs(np.diff(base)))


def test_velocity_scales_amplitude():
    p = PRESETS["pluck_lead"]
    soft = render(p, 60, 0.2, 40)
    loud = render(p, 60, 0.2, 110)
    assert np.abs(loud).max() > np.abs(soft).max()


def test_percussive_presets_produce_energy():
    for name in ("analog_kick", "noise_hat", "snare", "tom", "clap"):
        a = render(PRESETS[name], 38, 0.2, 100)
        assert _finite_bounded(a), name
        assert np.sqrt(np.mean(a ** 2)) > 1e-3, name   # real RMS energy


def test_render_event_uses_event_fields():
    ev = MusicEvent("note_on", 3, 64, 90, dur=0.25)
    a = render_event(PRESETS["warm_keys"], ev)
    b = render(PRESETS["warm_keys"], 64, 0.25, 90)
    assert np.array_equal(a, b)


def test_fxrack_zero_params_is_identity():
    rng = np.random.default_rng(0)
    buf = (rng.normal(0, 0.3, SR // 2)).astype(np.float32)
    out = FXRack().process(buf)
    assert out.shape == buf.shape
    assert np.allclose(out, buf, atol=1e-6)


def test_fxrack_each_effect_changes_and_stays_bounded():
    rng = np.random.default_rng(1)
    buf = (rng.normal(0, 0.3, SR // 2)).astype(np.float32)
    rack = FXRack()
    for kw in ({"reverb": 0.6}, {"delay": 0.6}, {"drive": 0.8},
               {"bitcrush": 0.7}):
        out = rack.process(buf, **kw)
        assert _finite_bounded(out), kw
        assert not np.allclose(out, buf, atol=1e-4), kw


def test_fxrack_is_deterministic():
    rng = np.random.default_rng(2)
    buf = (rng.normal(0, 0.3, SR // 4)).astype(np.float32)
    rack = FXRack()
    a = rack.process(buf, reverb=0.5, delay=0.4, drive=0.3, bitcrush=0.4)
    b = rack.process(buf, reverb=0.5, delay=0.4, drive=0.3, bitcrush=0.4)
    assert np.array_equal(a, b)


def test_presets_are_distinct():
    pitched = [n for n, p in PRESETS.items() if not p.percussive]
    sigs = {n: render(PRESETS[n], 55, 0.3, 100) for n in pitched}
    names = list(sigs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = sigs[names[i]], sigs[names[j]]
            m = min(a.size, b.size)
            assert not np.array_equal(a[:m], b[:m]), (names[i], names[j])
