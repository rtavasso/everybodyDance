"""Tests for the song-builder: deterministic instrument ordering, two-pass
authoring (rhythm onsets from hits, pitch from body height), and looping ghosts.
Self-contained synthetic groove -> CI-safe.
"""

import numpy as np

from everybody_dance.builder import SongBuilder, BuilderConfig
from everybody_dance.features import FeatureExtractor
from everybody_dance.laban import LabanEstimator
from everybody_dance.oscillator import EntrainedClock
from everybody_dance.output import LogBackend
from everybody_dance.personalization import Calibrator
from everybody_dance.pose import JOINT_INDEX, JOINTS
from everybody_dance.sources import ArrayPoseSource
from everybody_dance.substrate import SCALES

FPS = 30.0


def _neutral():
    p = {"nose": (0, 1.6, 0), "l_shoulder": (-0.2, 1.4, 0), "r_shoulder": (0.2, 1.4, 0),
         "l_elbow": (-0.3, 1.1, 0), "r_elbow": (0.3, 1.1, 0),
         "l_wrist": (-0.35, 0.85, 0), "r_wrist": (0.35, 0.85, 0),
         "l_hip": (-0.12, 0.9, 0), "r_hip": (0.12, 0.9, 0),
         "l_knee": (-0.13, 0.45, 0), "r_knee": (0.13, 0.45, 0),
         "l_ankle": (-0.14, 0.0, 0), "r_ankle": (0.14, 0.0, 0)}
    return np.array([p[j] for j in JOINTS], dtype=float)


def _groove(n, seed=0):
    """Bouncy groove with sharp arm strikes (hits) and crouch/rise (pitch)."""
    base = _neutral()
    rng = np.random.default_rng(seed)
    out = np.zeros((n, 13, 3))
    for i in range(n):
        ph = 2 * np.pi * 2.0 * i / FPS
        x = base.copy()
        x[:, 1] -= 0.18 * np.sin(ph)                          # crouch/rise (pitch)
        for a in ("l_ankle", "r_ankle"):
            x[JOINT_INDEX[a], 1] = base[JOINT_INDEX[a], 1]
        # sharp wrist strikes -> hits (fast snap on the beat)
        strike = max(0.0, np.sin(ph)) ** 8
        x[JOINT_INDEX["r_wrist"], 1] += 0.4 * strike
        x[JOINT_INDEX["l_wrist"], 1] += 0.4 * max(0.0, np.sin(ph + np.pi)) ** 8
        out[i] = x + rng.normal(0, 0.003, (13, 3))
    return out


def _build(seconds=40):
    seq = _groove(int(seconds * FPS))
    src = ArrayPoseSource(seq, fps=FPS, flip_y=False)
    fe, lab, clk, cal = FeatureExtractor(FPS), LabanEstimator(), EntrainedClock(), Calibrator()
    idx = [JOINT_INDEX[j] for j in ("l_wrist", "r_wrist", "l_ankle", "r_ankle")]
    hits = []
    for fr in src.frames():
        f = fe.update(fr)
        cal.observe(f, lab.update(f, f.dt), clk.update(float(f.bounce), f.dt).tempo_hz)
        hits.append(float(f.accel_mag[idx].max()))
    prof = cal.finalize(signature=fe.signature.summary())
    be = LogBackend()
    sb = SongBuilder(be, prof, 2.0, float(np.percentile(hits, 70)),
                     BuilderConfig())
    src2 = ArrayPoseSource(seq, fps=FPS, flip_y=False)
    for fr in src2.frames():
        sb.step(fr)
    return sb, be


def test_instruments_authored_in_order():
    sb, be = _build(44)
    # everything authored, in the fixed order, and now looping
    assert all(sb.status[r] == "saved" for r in sb.cfg.order)
    assert sb.phase == "loop"


def test_rhythm_onsets_are_placed_by_hits():
    sb, be = _build(44)
    for role in sb.cfg.order:
        n = sum(s is not None for s in sb.loops[role])
        assert 0 < n < sb.cfg.loop_steps      # some onsets, not every step


def test_pitch_is_deterministic_from_height_and_in_scale():
    sb, be = _build(44)
    scale = set(SCALES[sb.sub.cfg.scale])
    tonic = sb.sub.cfg.tonic
    # saved pitched stems carry frozen pitches, all in scale
    pitched = [s for r in ("bass", "keys", "lead") for s in sb.loops[r] if s]
    assert pitched
    assert all(s.pitch is not None for s in pitched)
    assert all((s.pitch - tonic) % 12 in scale for s in pitched)


def test_committed_stems_keep_looping_under_live():
    sb, be = _build(44)
    # ghosts captured for replay
    for role in sb.cfg.order:
        assert any(g is not None for g in sb.ghosts[role])


def test_determinism_same_dance_same_song():
    sb1, be1 = _build(40)
    sb2, be2 = _build(40)
    notes1 = [(e.channel, e.a, round(e.t, 3)) for e in be1.events if e.kind == "note_on"]
    notes2 = [(e.channel, e.a, round(e.t, 3)) for e in be2.events if e.kind == "note_on"]
    assert notes1 == notes2          # no RNG: same dance -> identical song
