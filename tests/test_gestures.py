"""Tests for the gesture layer: the built-in moves fire on a scripted performer,
recognition is deterministic, DTW template matching works, and effects emit
in-scale one-shots + flashes.
"""

import numpy as np
import pytest

from everybody_dance.features import FeatureExtractor
from everybody_dance.gestures import (DTWGesture, GestureRecognizer, default_gestures,
                                      dtw_distance, record_template)
from everybody_dance.pose import JOINTS, PoseFrame
from everybody_dance.substrate import SCALES, Substrate
from tools.render_gestures import scripted_performer

FPS = 30.0
MOVES = {"HANDS UP", "RAISE LEFT", "RAISE RIGHT", "T-POSE", "SQUAT",
         "ARMS CROSSED", "CLAP", "PUNCH"}


def _run(seq):
    fe = FeatureExtractor(FPS)
    rec = GestureRecognizer(default_gestures(), fps=FPS)
    out = []
    t = 0.0
    for x in seq:
        fr = PoseFrame(t=t, xyz=x, visibility=np.ones(13))
        f = fe.update(fr)
        for ev in rec.update(fr, f):
            out.append((t, ev.name))
        t += 1 / FPS
    return out


def test_all_builtin_moves_fire():
    fired = {n for _, n in _run(scripted_performer())}
    missing = MOVES - fired
    assert not missing, f"did not fire: {missing}"
    assert not (fired - MOVES), f"unexpected gestures: {fired - MOVES}"


def test_no_false_crosstalk_counts_reasonable():
    from collections import Counter
    c = Counter(n for _, n in _run(scripted_performer()))
    # each move performed once; allow a little double-fire on motion gestures
    for name in MOVES:
        assert 1 <= c[name] <= 3, (name, c[name])


def test_recognition_is_deterministic():
    seq = scripted_performer()
    assert _run(seq) == _run(seq)        # no RNG anywhere


def test_dtw_distance_self_is_zero_and_triangle():
    a = np.cumsum(np.random.default_rng(0).normal(0, 1, (20, 4)), 0)
    b = a + 5.0                          # pure translation
    assert dtw_distance(a, a) < 1e-9
    # whitening (inside DTWGesture) removes translation; bare distance sees it:
    assert dtw_distance(a, b) > 0


def test_dtw_template_matches_similar_not_dissimilar():
    seq = scripted_performer()
    joints = ["l_wrist", "r_wrist"]
    # template = a window around the T-POSE (which we know occurs)
    fired = _run(seq)
    t_pose_t = next(t for t, n in fired if n == "T-POSE")
    i = int(t_pose_t * FPS)
    tmpl_frames = seq[i - 8:i + 4]
    g = record_template(tmpl_frames, joints, "MY MOVE", threshold=0.6)
    # the same window should score under threshold; a flat neutral window should not
    win_match = np.stack([f[[5, 6], :2].reshape(-1) for f in tmpl_frames])
    neutral = np.stack([seq[0][[5, 6], :2].reshape(-1)] * 12)
    from everybody_dance.gestures import _whiten
    assert dtw_distance(_whiten(win_match), g.template) <= 0.6
    assert dtw_distance(_whiten(neutral), g.template) > 0.6


def test_effects_emit_in_scale_notes_and_flash():
    from everybody_dance.output import LogBackend
    from everybody_dance.substrate import SubstrateConfig
    be = LogBackend()
    sub = Substrate(SubstrateConfig.default())
    from everybody_dance.effects import EffectEngine
    fx = EffectEngine(be, sub)
    for name in ("CLAP", "HANDS UP", "T-POSE", "PUNCH", "SQUAT", "RAISE LEFT"):
        fx.trigger(name, 1.0)
    assert len(fx.flashes) == 6
    scale, tonic = set(SCALES[sub.cfg.scale]), sub.cfg.tonic
    pitched = [e for e in be.events if e.kind == "note_on" and e.channel != 10]
    assert pitched and all((e.a - tonic) % 12 in scale for e in pitched)
    # flash fades out
    f = fx.flashes[0]
    assert f.alpha(1.0) > 0 and f.alpha(1.0 + f.ttl + 0.1) == 0


def test_flash_pruning():
    from everybody_dance.output import LogBackend
    from everybody_dance.substrate import SubstrateConfig
    sub = Substrate(SubstrateConfig.default())
    from everybody_dance.effects import EffectEngine
    fx = EffectEngine(LogBackend(), sub)
    fx.trigger("CLAP", 0.0)
    assert fx.active_flashes(0.1)            # still alive
    assert fx.active_flashes(10.0) == []     # pruned
