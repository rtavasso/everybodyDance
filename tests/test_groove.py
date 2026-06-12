"""The groove backbone (everybody_dance.groove): latched tempo, curated
plans that repeat, the motif engine, and the rhythmic-coherence skill loop --
plus the studio-level proof that the grid is metronomic.

This is the regression net for THE quality property the instrument lives on:
a beat exists only if it repeats, on a grid that does not wobble.
"""

import sys

import numpy as np

from everybody_dance.groove import (MOTIF_TEMPLATES, STYLES, SWING,
                                    MotifWriter, RhythmCoherence, TempoLatch,
                                    bass_plan, drum_plan, keys_plan,
                                    texture_plan)
from everybody_dance.output import LogBackend
from everybody_dance.sources import ArrayPoseSource
from everybody_dance.studio import Studio
from everybody_dance.substrate import SCALES

from test_studio import FPS, _groove, _profile


# -- tempo latch ---------------------------------------------------------------

def test_latch_folds_into_band_and_quantizes():
    latch = TempoLatch(initial_hz=1.0)        # 60 -> folds to 120
    assert latch.bpm == 120.0
    assert latch.fold(40.0) == 80.0
    assert latch.fold(300.0) == 150.0
    assert 80.0 <= latch.fold(13.0) <= 160.0


def test_latch_ignores_jitter_relocks_on_sustained_shift():
    latch = TempoLatch(initial_hz=2.0)        # 120
    t = 0.0
    # jittery estimates around the current lock: no proposal.
    for k in range(120):
        t += 1 / 30.0
        latch.observe(2.0 + 0.05 * ((k % 5) - 2), t)
    assert latch.proposal() is None
    # a sustained, stable shift to 2.5 Hz (150): proposes ~150.
    for _ in range(120):
        t += 1 / 30.0
        latch.observe(2.5, t)
    prop = latch.proposal()
    assert prop is not None and abs(prop - 150.0) <= 2.0
    latch.relock(prop)
    assert latch.relocks == 1 and latch.bpm == prop


# -- plans ---------------------------------------------------------------------

def test_every_style_level_is_a_playable_pattern():
    for style, levels in STYLES.items():
        assert style in SWING
        assert len(levels) == 4
        for lvl, pat in enumerate(levels):
            assert 0 in pat["kick"], f"{style} L{lvl}: bar must start on a kick"
            for lane in ("kick", "snare", "hat", "ghost"):
                assert all(0 <= s < 16 for s in pat[lane])
            # density grows (weakly) with level.
            n = sum(len(pat[k]) for k in pat)
            if lvl:
                prev = levels[lvl - 1]
                assert n >= sum(len(prev[k]) for k in prev)


def test_full_levels_have_backbeat_and_plans_are_deterministic():
    for style in STYLES:
        assert STYLES[style][2]["snare"], f"{style} L2 needs a backbeat"
        assert drum_plan(style, 3) == drum_plan(style, 3)
    fill = drum_plan("backbeat", 1, fill=True)
    plain = drum_plan("backbeat", 1)
    assert sum(len(s) for s in fill) > sum(len(s) for s in plain)


def test_bass_locks_to_the_kick():
    for style in STYLES:
        for lvl in range(4):
            kicks = set(STYLES[style][lvl]["kick"])
            roots = {s for s, slots in enumerate(bass_plan(style, lvl))
                     for off, _, _ in slots if off == 0}
            assert kicks <= roots | kicks            # every kick has its root
            assert roots >= kicks


def test_keys_pad_low_stabs_high_and_texture_rotates():
    assert keys_plan("four_floor", 0) == [(0, "pad")]
    assert all(k == "stab" for _, k in keys_plan("four_floor", 3))
    a, b = texture_plan(2, bar_idx=0), texture_plan(2, bar_idx=4)
    assert a and b and a != b                        # the polyrhythm orbits
    assert texture_plan(0, 0) == []


# -- motif ----------------------------------------------------------------------

def _contour(h):
    return [h] * 32


def test_motif_persists_mutates_and_follows_height():
    w = MotifWriter(seed=3)
    w.maybe_compose(2, _contour(0.2), bar_idx=0)
    low = list(w.slots)
    assert len(low) >= 4
    w.maybe_compose(2, _contour(0.2), bar_idx=1)     # mid-phrase: unchanged
    assert w.slots == low
    w.maybe_compose(2, _contour(0.9), bar_idx=4)     # phrase: ONE slot mutates
    assert sum(1 for a, b in zip(low, w.slots) if a != b) <= 1
    w2 = MotifWriter(seed=3)
    w2.maybe_compose(2, _contour(0.9), bar_idx=0)    # high body -> higher line
    assert (np.mean([o for _, o, _ in w2.slots])
            > np.mean([o for _, o, _ in low]))


def test_motif_strong_slots_are_chord_tones_and_steps_are_singable():
    w = MotifWriter(seed=1)
    w.maybe_compose(3, [0.1 + 0.8 * (k / 31) for k in range(32)], bar_idx=0)
    prev = None
    for slot, off, strong in w.slots:
        if strong:
            assert off in (0, 2, 4, 7)
        if prev is not None:
            assert abs(off - prev) <= 3              # stepwise: no wild leaps
        prev = off
    plan = w.plan(chord_root_degree=3)
    assert [d - 3 for _, d, _ in plan] == [o for _, o, _ in w.slots]


# -- coherence -------------------------------------------------------------------

def _run_pulse(intervals, beat_s=0.5):
    c = RhythmCoherence()
    t, dt = 0.0, 1 / 30.0
    times, k, nxt = [], 0, intervals[0]
    while t < sum(intervals):
        # triangular bounce pulse around each scheduled peak
        phase = (t - (nxt - intervals[k % len(intervals)])) \
            / intervals[k % len(intervals)]
        b = 1.0 - abs(2 * ((t / intervals[k % len(intervals)]) % 1.0) - 1.0)
        c.update(0.2 * b, t, beat_s, dt)
        t += dt
        if t >= nxt:
            k += 1
            nxt += intervals[k % len(intervals)]
    return c


def test_coherence_high_on_the_beat_low_when_erratic():
    on = _run_pulse([0.5] * 40)                      # pulsing AT the beat
    # deterministic-but-erratic intervals (not multiples of the beat)
    err = _run_pulse([0.31, 0.74, 0.52, 0.43, 0.66, 0.38] * 7)
    assert on.score > err.score + 0.15
    assert on.score > 0.6


def test_coherence_tempo_estimate_matches_pulse():
    c = _run_pulse([0.5] * 40)
    assert c.tempo_hz is not None
    assert abs(c.tempo_hz - 2.0) < 0.15


def test_gate_level_caps_when_loose():
    c = RhythmCoherence()
    c.score = 0.2
    assert c.gate_level(3) == 1
    c.score = 0.5
    assert c.gate_level(3) == 2
    c.score = 0.8
    assert c.gate_level(3) == 3


# -- studio-level: the grid is metronomic ------------------------------------------

def test_studio_grid_is_metronomic_and_bars_repeat():
    n = int(20 * FPS)
    seq = _groove(n)
    be = LogBackend()
    st = Studio(be, _profile(seq))
    for fr in ArrayPoseSource(seq, fps=FPS, flip_y=False).frames():
        st.step(fr, [])
    st.panic()
    # after the lock settles, kick IOIs are exact multiples of the step.
    step_s = st.latch.beat_s / 4
    kicks = [e.t for e in be.events
             if e.kind == "note_on" and e.channel == 10 and e.a == 36
             and e.t > 10.0]
    assert len(kicks) >= 6
    for ioi in np.diff(kicks):
        assert abs(ioi / step_s - round(ioi / step_s)) < 0.02, ioi
    # the backbeat exists and the key stays in scale per the key log.
    snares = [e for e in be.events if e.kind == "note_on" and e.a == 38
              and e.channel == 10]
    assert snares
    deg = set(SCALES[st.sub.cfg.scale])
    for e in be.events:
        if e.kind == "note_on" and e.channel != 10 and e.t > 10.0:
            assert (e.a - st.sub.cfg.tonic) % 12 in deg
