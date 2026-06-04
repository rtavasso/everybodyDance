"""Tests for the live-sandbox plumbing: shared voices, the real-time mixer (no
audio device needed), calibration folding, and the live-frame composition.
GUI/audio/camera themselves can't run in CI, so we test the pure logic underneath.
"""

import numpy as np
import pytest

from everybody_dance.calib import fold_tempo
from everybody_dance.output import MusicEvent
from everybody_dance.rtaudio import RealtimeSynth
from everybody_dance.voices import SR, midi_hz, render_note


def test_voices_are_finite_bounded_and_deterministic():
    for ch, p, d, v in [(10, 36, 0.1, 100), (10, 42, 0.1, 90), (10, 38, 0.2, 80),
                        (1, 40, 0.5, 100), (2, 60, 1.0, 80), (3, 72, 0.3, 90)]:
        a = render_note(ch, p, d, v)
        b = render_note(ch, p, d, v)
        assert a.size > 0 and np.all(np.isfinite(a))
        assert np.abs(a).max() < 5.0
        assert np.array_equal(a, b)              # no per-call RNG drift


def test_midi_hz_a440():
    assert abs(midi_hz(69) - 440.0) < 1e-6


def test_rtmixer_mixes_and_drains_without_a_device():
    rt = RealtimeSynth()                          # no PortAudio in CI -> ok False
    rt.ok = True                                  # exercise the pure mixer anyway
    rt.send(MusicEvent("note_on", 1, 60, 100, dur=0.05, t=0))
    assert len(rt._voices) == 1
    total = 0
    block = rt._mix(512)
    assert block.shape == (512,) and np.all(np.isfinite(block))
    assert np.abs(block).max() <= 1.0            # soft-clipped
    # drain to completion
    for _ in range(int(0.05 * SR / 512) + 2):
        rt._mix(512)
    assert rt._voices == []                       # finished voice removed


def test_rtmixer_polyphony_cap():
    rt = RealtimeSynth()
    rt.ok = True
    for _ in range(100):
        rt.send(MusicEvent("note_on", 3, 72, 90, dur=1.0, t=0))
    assert len(rt._voices) <= 48
    rt.panic()
    assert rt._voices == []


def test_disabled_backend_is_silent_noop():
    rt = RealtimeSynth()
    rt.ok = False
    rt.send(MusicEvent("note_on", 1, 60, 100, dur=0.2, t=0))   # must not raise
    assert rt._voices == []


def test_fold_tempo_into_band():
    assert 1.0 <= fold_tempo(0.6) <= 2.3
    assert 1.0 <= fold_tempo(3.9) <= 2.3
    assert abs(fold_tempo(2.0) - 2.0) < 1e-9
    assert fold_tempo(0.0) == 2.0


def test_compose_live_frame_shape():
    cv2 = pytest.importorskip("cv2")              # viz needs opencv
    from everybody_dance.builder import BuilderConfig, UIState
    from everybody_dance.viz import compose_live
    cfg = BuilderConfig()
    ui = UIState(order=list(cfg.order),
                 status={r: "saved" for r in cfg.order}, active="lead",
                 phase="rhythm", bars_left=1, playhead=4,
                 onsets={r: [i % 4 == 0 for i in range(cfg.loop_steps)] for r in cfg.order},
                 pitch01=0.6, live_xyz=np.zeros((13, 3)),
                 ghosts={r: np.zeros((13, 3)) for r in cfg.order}, bpm=120.0)
    cam = np.zeros((480, 640, 3), np.uint8)
    canvas = compose_live(cam, ui, {"fps": 30, "infer_ms": 20, "audio": True,
                                    "landmarks_px": None, "note": "x"}, cfg.loop_steps)
    assert canvas.shape == (720, 1280, 3)
