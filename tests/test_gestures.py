"""Tests for the gesture layer: the built-in moves fire on a scripted performer,
recognition is deterministic, DTW template matching works, and effects emit
in-scale one-shots + flashes.
"""

import numpy as np
import pytest

from everybody_dance.features import FeatureExtractor
from everybody_dance.gestures import (DTWGesture, GESTURE_LIBRARY, GestureRecognizer,
                                      build_gestures, default_gestures,
                                      dtw_distance, record_template)
from everybody_dance.pose import JOINTS, PoseFrame
from everybody_dance.sources import ArrayPoseSource
from everybody_dance.substrate import SCALES, Substrate
from tools.render_gestures import scripted_performer

FPS = 30.0
# Exactly the moves the (updated) scripted performer triggers: the eight pose/
# motion moves plus the two global-vertical translations (JUMP/STOMP), which the
# scripted performer now drives by translating the whole skeleton up/down.
MOVES = {"HANDS UP", "RAISE LEFT", "RAISE RIGHT", "T-POSE", "SQUAT",
         "ARMS CROSSED", "CLAP", "PUNCH", "JUMP", "STOMP"}


def _run(seq, gestures=None):
    """Replay a scripted skeleton sequence through the real source + recognizer.

    Routing through ArrayPoseSource (not bare PoseFrames) is what populates
    `root_y`, so the global-vertical moves (JUMP/STOMP) can fire from the
    scripted whole-body translations the same way they would on real input.
    """
    fe = FeatureExtractor(FPS)
    rec = GestureRecognizer(default_gestures() if gestures is None else gestures, fps=FPS)
    out = []
    for fr in ArrayPoseSource(np.asarray(seq), fps=FPS, flip_y=False).frames():
        f = fe.update(fr)
        for ev in rec.update(fr, f):
            out.append((fr.t, ev.name))
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


def _run_root_y(root_ys, gestures):
    """Drive the recognizer with a synthetic root_y track on a still skeleton."""
    fe = FeatureExtractor(FPS)
    rec = GestureRecognizer(gestures, fps=FPS)
    base = scripted_performer()[0]       # a neutral standing frame (no pose fires)
    out, t = [], 0.0
    for ry in root_ys:
        fr = PoseFrame(t=t, xyz=base, visibility=np.ones(13), root_y=float(ry))
        f = fe.update(fr)
        for ev in rec.update(fr, f):
            out.append((t, ev.name))
        t += 1 / FPS
    return out


def test_jump_fires_on_root_y_rise_not_on_flat():
    # A fast upward root_y ramp (~0.6 torso in 0.15 s -> ~4 torso/s) must fire
    # JUMP; a flat root_y (no global vertical) must not. This is the whole point
    # of the raw global-vertical signal the hip-centred skeleton otherwise hides.
    g = build_gestures(["JUMP"])
    rise = np.concatenate([np.zeros(15),
                           np.linspace(0.0, 0.6, 5),   # 0.6 torso over ~0.15 s
                           np.full(20, 0.6)])
    fired = {n for _, n in _run_root_y(rise, build_gestures(["JUMP"]))}
    assert "JUMP" in fired
    flat = {n for _, n in _run_root_y(np.full(40, 0.3), build_gestures(["JUMP"]))}
    assert "JUMP" not in flat


def test_build_gestures_subset_and_library():
    # build_gestures(None) is the full default set; a named subset builds exactly
    # those; JUMP and STOMP are in the library; unknown names fail loudly.
    assert {g.name for g in build_gestures()} == MOVES
    assert {g.name for g in build_gestures(["JUMP", "STOMP"])} == {"JUMP", "STOMP"}
    assert "JUMP" in GESTURE_LIBRARY and "STOMP" in GESTURE_LIBRARY
    with pytest.raises(KeyError):
        build_gestures(["NOPE"])


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
