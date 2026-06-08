"""Tests for the observability metrics (deterministic, no cv2/API needed)."""

import numpy as np

from everybody_dance import metrics as M
from everybody_dance.output import MusicEvent


def test_coupling_high_when_music_tracks_movement():
    move = {"energy": np.array([0.0, 1, 2, 3, 4, 3, 2, 1])}
    music = {"density": move["energy"] * 2 + 1}      # perfectly correlated
    c = M.coupling(move, music)
    assert c["score"] > 0.95
    assert c["dead_zone"] == 0.0


def test_coupling_low_when_uncorrelated():
    rng = np.random.default_rng(0)
    move = {"energy": rng.normal(0, 1, 64)}
    music = {"density": rng.normal(0, 1, 64)}
    assert M.coupling(move, music)["score"] < 0.4


def test_coupling_flags_dead_zone():
    # high movement but zero music in those bins
    move = {"energy": np.array([0.0, 0, 0, 5, 6, 7])}
    music = {"density": np.array([2.0, 3, 2, 0, 0, 0])}
    assert M.coupling(move, music)["dead_zone"] == 1.0


def test_musicality_in_scale():
    # C major-ish: build events on scale tonic 60, major scale
    ons = [MusicEvent("note_on", 3, 60 + p, 100, t=i * 0.1)
           for i, p in enumerate([0, 2, 4, 5, 7])]      # all in major
    m = M.musicality(ons, tonic=60, scale="major", minutes=0.5)
    assert m["in_scale_pct"] == 100.0 and m["notes"] == 5


def test_recognition_recall_precision_latency():
    labels = [(1.0, "CLAP"), (2.0, "JUMP")]
    fired = [(1.05, "CLAP"), (5.0, "PUNCH")]          # 1 hit, 1 miss, 1 FP
    r = M.recognition(fired, labels)
    assert r["recall"] == 0.5 and r["tp"] == 1 and r["fp"] == 1
    assert r["precision"] == 0.5
    assert abs(r["median_latency_ms"] - 50.0) < 1.0


def test_recognition_unlabelled_reports_counts():
    r = M.recognition([(1.0, "CLAP"), (2.0, "CLAP")], None)
    assert r["labelled"] is False and r["fires"]["CLAP"] == 2


def test_timing_headroom():
    t = M.timing([1.0] * 100, fps=30)                 # 1ms compute, 33ms budget
    assert t["headroom_pct"] > 90


def test_slo_returns_plain_bools():
    flat = {"coupling.score": 0.5, "coupling.dead_zone": 0.1,
            "musicality.in_scale_pct": 100.0, "timing.headroom_pct": 50.0}
    slo = M.check_slo(flat)
    assert all(isinstance(v, bool) for v in slo.values())
    assert all(slo.values())
    import json
    json.dumps(slo)                                   # must be serialisable


def test_slo_supports_strict_operators():
    # liveliness.density_std uses a strict ">"; 0.0 must fail, >0 must pass.
    assert M.check_slo({"liveliness.density_std": 0.0})["liveliness.density_std"] is False
    assert M.check_slo({"liveliness.density_std": 0.5})["liveliness.density_std"] is True


def test_liveliness_varies_and_finds_silence():
    # 4 onsets in the first second, then an 8s gap, then one more.
    ons = [MusicEvent("note_on", 3, 60, 100, t=t) for t in (0.1, 0.3, 0.6, 0.9)]
    ons.append(MusicEvent("note_on", 3, 62, 100, t=9.5))
    lv = M.liveliness(ons, dur=10.0, window_s=1.0)
    assert lv["density_std"] > 0.0                    # not a flat wall
    assert lv["longest_silence_s"] >= 8.0             # the gap is found
    # empty -> no variation, whole duration silent
    assert M.liveliness([], dur=5.0)["density_std"] == 0.0


def test_gesture_spam_rate_per_minute():
    fired = [(t, "CLAP") for t in (0.0, 1.0, 2.0)] + [(0.5, "JUMP")]
    g = M.gesture_spam(fired, dur=60.0)
    assert g["max_per_min"] == 3.0 and g["per_gesture_per_min"]["CLAP"] == 3.0
    assert g["total"] == 4
    assert M.gesture_spam([], dur=60.0)["max_per_min"] == 0.0


def test_phantom_flagged_when_sound_during_stillness():
    # an energy gradient whose lowest bin sits below the 10th percentile, with
    # music playing in that (near-still) bin -> a phantom.
    move = {"energy": np.arange(10.0)}                 # 0..9; p10 ~ 0.9
    music = {"density": np.full(10, 2.0)}              # never stops, even at rest
    c = M.coupling(move, music)
    assert c["phantom"] > 0.5
    # SLO should mark phantom as failing (> 0.25)
    assert M.check_slo({"coupling.phantom": c["phantom"]})["coupling.phantom"] is False
