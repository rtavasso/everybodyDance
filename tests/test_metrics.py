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
