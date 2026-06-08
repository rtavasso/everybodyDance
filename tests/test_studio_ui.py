"""Tests for everybody_dance.studio_ui — the Just-Dance-inspired compositor.

Guards the cv2 import with importorskip so headless CI skips cleanly.
"""

from __future__ import annotations

import pytest

cv2 = pytest.importorskip("cv2")  # skip entire module if cv2 not installed
import numpy as np

from everybody_dance.studio_types import (
    FxConfig,
    MovePrompt,
    PitchConfig,
    RhythmConfig,
    StemConfig,
    StemState,
    StudioUIState,
    TimbreConfig,
    default_stem_configs,
)
from everybody_dance.effects import Flash
from everybody_dance.studio_ui import attract_screen, compose_studio

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

W, H = 1280, 720


def _make_stem(name: str, channel: int, color=(80, 120, 255), n_steps: int = 16,
               instrument: str = "kick") -> StemState:
    """Build a StemState with a sparse loop (every 4th step filled)."""
    from everybody_dance.studio_types import Note
    cfg = StemConfig(
        name=name,
        channel=channel,
        color=color,
        timbre=TimbreConfig(instrument=instrument, cutoff=0.7),
        fx=FxConfig(delay_send=0.2, reverb_send=0.1),
        rhythm=RhythmConfig(steps=n_steps, bars=1),
        pitch=PitchConfig(mode="fixed"),
    )
    loop: list = [None] * n_steps
    for s in range(0, n_steps, 4):
        loop[s] = Note(step=s, pitch=60, vel=100)
    return StemState(config=cfg, loop=loop)


def _populated_state() -> StudioUIState:
    """A realistic populated StudioUIState for rendering tests."""
    stem_a = _make_stem("drums", 10, color=(80, 120, 255), instrument="kick")
    stem_b = _make_stem("bass", 1, color=(80, 220, 120), instrument="bass")
    stem_b.muted = True
    stem_b.solo = False

    stem_c = _make_stem("keys", 2, color=(240, 180, 70), instrument="pad")
    stem_c.recording = True

    prompts = [
        MovePrompt(name="CLAP", due_t=1.5, lead_s=2.0, hit=None),
        MovePrompt(name="HANDS UP", due_t=3.0, lead_s=2.0, hit=True),
        MovePrompt(name="SQUAT", due_t=0.2, lead_s=2.0, hit=False),
    ]

    flash = Flash(name="CLAP", color=(255, 255, 255), t0=0.0, ttl=0.45, big=False)

    live_xyz = np.zeros((13, 3), dtype=np.float32)

    return StudioUIState(
        stems=[stem_a, stem_b, stem_c],
        active_stem=0,
        phase="perform",
        playhead=4,
        bpm=128.0,
        pitch01=0.65,
        flashes=[flash],
        prompts=prompts,
        score=12345,
        combo=7,
        rating="PERFECT",
        meters={"energy": 0.8, "cutoff": 0.5, "pitch": 0.3},
        live_xyz=live_xyz,
        present=True,
        bar=2,
        bars_left=4,
    )


def _idle_state() -> StudioUIState:
    """Minimal empty / idle state — dancer not present, no stems."""
    return StudioUIState(
        stems=[],
        active_stem=0,
        phase="idle",
        playhead=0,
        bpm=100.0,
        pitch01=0.5,
        flashes=[],
        prompts=[],
        score=0,
        combo=0,
        rating="",
        meters={},
        live_xyz=None,
        present=False,
        bar=0,
        bars_left=0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestComposeStudio:
    def test_shape_dtype_populated(self):
        state = _populated_state()
        canvas = compose_studio(state, W=W, H=H, t=0.5)
        assert canvas.shape == (H, W, 3), f"expected ({H},{W},3), got {canvas.shape}"
        assert canvas.dtype == np.uint8

    def test_shape_dtype_idle(self):
        """Empty/idle state (no stems, not present, no xyz) must not crash."""
        state = _idle_state()
        canvas = compose_studio(state, W=W, H=H, t=0.0)
        assert canvas.shape == (H, W, 3)
        assert canvas.dtype == np.uint8

    def test_not_all_black_populated(self):
        """Populated state should produce a non-trivial canvas."""
        state = _populated_state()
        canvas = compose_studio(state, W=W, H=H, t=0.1)
        assert canvas.max() > 30, "canvas is suspiciously dark"

    def test_with_camera_bgr(self):
        """Passing a camera frame should not crash or change output shape."""
        state = _populated_state()
        cam = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        canvas = compose_studio(state, W=W, H=H, camera_bgr=cam, t=1.0)
        assert canvas.shape == (H, W, 3)
        assert canvas.dtype == np.uint8

    def test_various_t_values(self):
        """Deterministic at several t values — no crashes, correct shape."""
        state = _populated_state()
        for t in (0.0, 0.25, 0.9, 2.0, 10.0):
            canvas = compose_studio(state, W=W, H=H, t=t)
            assert canvas.shape == (H, W, 3), f"failed at t={t}"

    def test_empty_prompts_and_flashes(self):
        """State with stems but no prompts/flashes/meters."""
        stem = _make_stem("lead", 3, color=(200, 100, 240), instrument="pluck")
        state = StudioUIState(
            stems=[stem],
            active_stem=0,
            phase="record_rhythm",
            playhead=0,
            bpm=90.0,
            pitch01=0.4,
            flashes=[],
            prompts=[],
            score=0,
            combo=0,
            rating="",
            meters={},
            live_xyz=np.zeros((13, 3), dtype=np.float32),
            present=True,
            bar=0,
            bars_left=8,
        )
        canvas = compose_studio(state, W=W, H=H, t=0.0)
        assert canvas.shape == (H, W, 3)
        assert canvas.dtype == np.uint8

    def test_miss_rating(self):
        state = _populated_state()
        state.rating = "MISS"
        state.combo = 0
        canvas = compose_studio(state, W=W, H=H, t=0.05)
        assert canvas.shape == (H, W, 3)

    def test_good_rating(self):
        state = _populated_state()
        state.rating = "GOOD"
        canvas = compose_studio(state, W=W, H=H, t=0.3)
        assert canvas.shape == (H, W, 3)

    def test_solo_stem(self):
        stem = _make_stem("drums", 10)
        stem.solo = True
        state = StudioUIState(
            stems=[stem],
            active_stem=0,
            phase="perform",
            playhead=2,
            bpm=120.0,
            pitch01=0.5,
            flashes=[],
            prompts=[],
            score=0,
            combo=0,
            rating="",
            meters={"energy": 0.6},
            live_xyz=None,
            present=False,
            bar=1,
            bars_left=0,
        )
        canvas = compose_studio(state, W=W, H=H, t=0.0)
        assert canvas.shape == (H, W, 3)

    def test_four_stems_default_configs(self):
        """All four default stem configs as StemStates."""
        stems = [
            StemState(config=cfg, loop=[None] * cfg.rhythm.total_steps)
            for cfg in default_stem_configs()
        ]
        state = StudioUIState(
            stems=stems,
            active_stem=2,
            phase="perform",
            playhead=8,
            bpm=110.0,
            pitch01=0.7,
            flashes=[],
            prompts=[MovePrompt(name="T-POSE", due_t=2.0, lead_s=2.0)],
            score=9999,
            combo=3,
            rating="",
            meters={"energy": 0.5, "pitch": 0.8},
            live_xyz=np.zeros((13, 3), dtype=np.float32),
            present=True,
            bar=0,
            bars_left=2,
        )
        canvas = compose_studio(state, W=W, H=H, t=1.23)
        assert canvas.shape == (H, W, 3)
        assert canvas.dtype == np.uint8

    def test_non_default_resolution(self):
        """compose_studio must work with non-default W/H."""
        state = _idle_state()
        canvas = compose_studio(state, W=960, H=540, t=0.0)
        assert canvas.shape == (540, 960, 3)
        assert canvas.dtype == np.uint8


class TestAttractScreen:
    def test_shape_dtype(self):
        canvas = attract_screen(W=W, H=H, t=0.0)
        assert canvas.shape == (H, W, 3)
        assert canvas.dtype == np.uint8

    def test_non_default_resolution(self):
        canvas = attract_screen(W=800, H=600, t=2.5)
        assert canvas.shape == (600, 800, 3)
        assert canvas.dtype == np.uint8

    def test_various_t_values(self):
        for t in (0.0, 1.0, 5.0, 99.9):
            canvas = attract_screen(W=W, H=H, t=t)
            assert canvas.shape == (H, W, 3), f"failed at t={t}"

    def test_animated_not_static(self):
        """Two different t values should produce different canvases."""
        c0 = attract_screen(W=W, H=H, t=0.0)
        c1 = attract_screen(W=W, H=H, t=0.5)
        assert not np.array_equal(c0, c1), "attract_screen appears static (no animation)"
