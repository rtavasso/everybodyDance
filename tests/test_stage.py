"""Headless tests for the gallery visual identity (everybody_dance.stage).

CI-safe: NO camera, NO mediapipe, NO audio. We build synthetic StudioUI/StemUI
snapshots directly and assert the rendered canvases are well-formed, non-trivial
images, and that the no-person / empty-flashes / empty-onsets frames don't crash.
Deterministic throughout.
"""

import numpy as np

from everybody_dance.effects import Flash
from everybody_dance.pose import JOINTS, JOINT_INDEX
from everybody_dance.stage import Stage, compose_stage, render_attract
from everybody_dance.stems import STEM_COLOR, STEM_ORDER
from everybody_dance.studio import StemUI, StudioUI

W, H = 1280, 720


def _skeleton():
    """A simple standing skeleton in torso-normalised space (y up)."""
    p = {"nose": (0.0, 1.55, 0.0),
         "l_shoulder": (-0.25, 1.15, 0.0), "r_shoulder": (0.25, 1.15, 0.0),
         "l_elbow": (-0.34, 0.65, 0.0), "r_elbow": (0.34, 0.65, 0.0),
         "l_wrist": (-0.40, 0.15, 0.0), "r_wrist": (0.40, 0.15, 0.0),
         "l_hip": (-0.16, 0.0, 0.0), "r_hip": (0.16, 0.0, 0.0),
         "l_knee": (-0.18, -0.95, 0.0), "r_knee": (0.18, -0.95, 0.0),
         "l_ankle": (-0.19, -1.9, 0.0), "r_ankle": (0.19, -1.9, 0.0)}
    return np.array([p[j] for j in JOINTS], dtype=float)


def _stems(active=True, onsets=True, loop_steps=32):
    out = []
    for k, name in enumerate(STEM_ORDER):
        ons = [(i % 4 == k % 4) for i in range(loop_steps)] if onsets \
            else [False] * loop_steps
        out.append(StemUI(
            name=name, color=STEM_COLOR[name], level=0.3 + 0.1 * k,
            active=active, looping=(name == "bass"), recording=(name == "keys"),
            muted=(name == "texture"), density=0.5, cutoff=0.6,
            onsets=ons, timbre="analog_kick"))
    return out


def _perform_ui(present=True, flashes=None, onsets=True, xyz=None):
    return StudioUI(
        t=1.0, phase="perform", bpm=120.0, present=present, energy=0.7,
        live_xyz=_skeleton() if xyz is None else xyz,
        stems=_stems(onsets=onsets), playhead=5, loop_steps=32,
        flashes=flashes if flashes is not None else [],
        prompt="JUMP -> fill", mood="minor", beat=True)


def _attract_ui():
    return StudioUI(
        t=2.0, phase="attract", bpm=0.0, present=False, energy=0.0,
        live_xyz=np.zeros((len(JOINTS), 3)), stems=_stems(active=False),
        playhead=0, loop_steps=32, flashes=[], prompt="step in to play",
        mood="major", beat=False)


def _is_image(img):
    assert isinstance(img, np.ndarray)
    assert img.shape == (H, W, 3)
    assert img.dtype == np.uint8


def _nontrivial(img):
    # not a flat single-colour frame -- there is actual drawing in it
    assert int(img.max()) - int(img.min()) > 20
    assert len(np.unique(img.reshape(-1, 3), axis=0)) > 50


# -- perform frame ---------------------------------------------------------

def test_compose_stage_perform_is_well_formed_image():
    flash = Flash("HANDS UP", (90, 240, 255), t0=1.0, ttl=0.5, big=True)
    ui = _perform_ui(flashes=[flash])
    perf = {"fps": 30.0, "infer_ms": 12.0, "audio": True, "note": "audio ON"}
    img = compose_stage(None, ui, perf, W=W, H=H)
    _is_image(img)
    _nontrivial(img)


def test_compose_stage_with_camera_background():
    cam = np.full((480, 640, 3), 120, np.uint8)
    img = compose_stage(cam, _perform_ui(), {"fps": 24}, W=W, H=H)
    _is_image(img)
    _nontrivial(img)


# -- attract frame ---------------------------------------------------------

def test_render_attract_is_well_formed_and_animated():
    a = render_attract(0.0, W=W, H=H)
    b = render_attract(1.7, W=W, H=H)
    _is_image(a)
    _is_image(b)
    _nontrivial(a)
    # animates purely from t -> a different phase yields a different frame
    assert not np.array_equal(a, b)


def test_compose_stage_attract_phase_routes_to_attract():
    img = compose_stage(None, _attract_ui(), {})
    _is_image(img)
    _nontrivial(img)


# -- robustness (the no-person frame) --------------------------------------

def test_compose_stage_robust_to_zero_skeleton_and_empty_state():
    ui = StudioUI(
        t=0.0, phase="perform", bpm=0.0, present=False, energy=0.0,
        live_xyz=np.zeros((len(JOINTS), 3)),
        stems=_stems(active=False, onsets=False),
        playhead=0, loop_steps=32, flashes=[], prompt=None, mood="weird_scale",
        beat=False)
    img = compose_stage(None, ui, {})       # None camera, zero skeleton, no flashes
    _is_image(img)


def test_compose_stage_robust_to_empty_stems():
    ui = _perform_ui()
    ui.stems = []
    img = compose_stage(None, ui, {})
    _is_image(img)


def test_compose_stage_default_size():
    img = compose_stage(None, _perform_ui(), {})
    assert img.shape == (720, 1280, 3)


# -- determinism -----------------------------------------------------------

def test_attract_is_deterministic_in_t():
    a1 = Stage(W, H).attract(0.5)
    a2 = Stage(W, H).attract(0.5)
    assert np.array_equal(a1, a2)


# -- import contract -------------------------------------------------------

def test_stage_and_studio_live_import_clean():
    import importlib

    import everybody_dance.stage as stage_mod
    import tools.studio_live as live_mod

    importlib.reload(stage_mod)            # imports without a camera
    assert hasattr(stage_mod, "compose_stage")
    assert hasattr(stage_mod, "render_attract")
    assert hasattr(stage_mod, "Stage")
    assert hasattr(live_mod, "main")
    assert hasattr(live_mod, "landmarks_to_arrays")
