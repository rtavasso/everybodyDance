"""Just-Dance-inspired UI compositor for the everybodyDance studio.

compose_studio() blends the live dancer, stem rack, move-prompt lane, score HUD,
side meters, and gesture flashes onto a single BGR canvas. Deterministic given
(state, t). cv2 is imported at module level; guard with pytest.importorskip in CI.
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from everybody_dance import viz
from everybody_dance.effects import Flash
from everybody_dance.studio_types import StudioUIState, StemState

# ---------------------------------------------------------------------------
# Layout constants (1280 x 720 grid)
# ---------------------------------------------------------------------------
_BG = 18                        # very dark background
_HDR_H = 70                     # header height
_STEM_ROW_H = 52                # height per stem row in the rack
_PROMPT_H = 80                  # move-prompt lane height
_METER_W = 52                   # side-meter column width
_HIT_LINE_X_FRAC = 0.18         # hit-line as fraction of prompt lane width
_RATING_COLORS = {
    "PERFECT": (80, 255, 160),
    "GOOD": (80, 200, 255),
    "MISS": (60, 60, 220),
}
_COMBO_COLORS = [
    (200, 200, 200), (80, 220, 120), (240, 180, 70),
    (200, 100, 240), (80, 200, 255), (255, 180, 80),
]
_MOVE_COLORS = {
    "default": (180, 180, 180),
    "CLAP": (255, 255, 255),
    "PUNCH": (90, 90, 255),
    "HANDS UP": (90, 240, 255),
    "T-POSE": (255, 120, 220),
    "SQUAT": (255, 200, 80),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compose_studio(
    state: StudioUIState,
    W: int = 1280,
    H: int = 720,
    camera_bgr=None,
    t: float = 0.0,
) -> np.ndarray:
    """Return a BGR uint8 canvas compositing the full studio UI."""
    canvas = np.full((H, W, 3), _BG, np.uint8)

    stem_rack_h = _STEM_ROW_H * max(len(state.stems), 1)
    prompt_y = H - stem_rack_h - _PROMPT_H
    stage_y = _HDR_H
    stage_h = prompt_y - stage_y

    # Meter strip on the right
    meter_x = W - _METER_W
    stage_w = meter_x

    _draw_header(canvas, state, W, t)
    _draw_center_stage(canvas, state, stage_x=0, stage_y=stage_y,
                       stage_w=stage_w, stage_h=stage_h,
                       camera_bgr=camera_bgr, t=t)
    _draw_prompt_lane(canvas, state, lane_x=0, lane_y=prompt_y,
                      lane_w=stage_w, lane_h=_PROMPT_H, t=t)
    _draw_stem_rack(canvas, state, rack_x=0, rack_y=H - stem_rack_h,
                    rack_w=W, row_h=_STEM_ROW_H)
    _draw_side_meters(canvas, state, mx=meter_x, my=stage_y,
                      mh=stage_h + _PROMPT_H)

    # Gesture flashes drawn last so they sit on top
    viz.draw_flashes(canvas, state.flashes, t)
    return canvas


def attract_screen(W: int = 1280, H: int = 720, t: float = 0.0) -> np.ndarray:
    """Looping idle / attract screen, animated by t."""
    canvas = np.full((H, W, 3), _BG, np.uint8)
    pulse = 0.5 + 0.5 * math.sin(t * 2.0)
    ring_col = (
        int(60 + 120 * pulse),
        int(40 + 80 * pulse),
        int(80 + 160 * pulse),
    )
    cx, cy = W // 2, H // 2
    # Animated concentric rings
    for r in range(3):
        phase = t * 1.2 + r * 1.1
        radius = int(120 + 60 * math.sin(phase) + r * 70)
        alpha = max(0.0, 0.6 - r * 0.15) * (0.5 + 0.5 * math.sin(phase))
        ov = canvas.copy()
        cv2.circle(ov, (cx, cy), radius, ring_col, 3, cv2.LINE_AA)
        cv2.addWeighted(ov, alpha, canvas, 1.0 - alpha, 0, canvas)

    # Title
    title = "everybodyDance"
    (tw, th), _ = cv2.getTextSize(title, viz.FONT, 2.0, 4)
    cv2.putText(canvas, title, ((W - tw) // 2, cy - 60),
                viz.FONT, 2.0, (0, 0, 0), 8, cv2.LINE_AA)
    cv2.putText(canvas, title, ((W - tw) // 2, cy - 60),
                viz.FONT, 2.0, (220, 200, 255), 4, cv2.LINE_AA)

    # Animated invite text — blink
    blink_a = 0.5 + 0.5 * math.sin(t * 3.0)
    invite = "step in to play"
    (iw, _), _ = cv2.getTextSize(invite, viz.FONT, 1.0, 2)
    col = tuple(int(c * blink_a) for c in (90, 240, 200))
    cv2.putText(canvas, invite, ((W - iw) // 2, cy + 40),
                viz.FONT, 1.0, col, 2, cv2.LINE_AA)

    # Silhouette pose hint (static skeleton-like figure)
    _draw_idle_figure(canvas, cx, cy + 120, t)

    # Bottom tag
    tag = "dance  .  build  .  perform"
    (tw2, _), _ = cv2.getTextSize(tag, viz.FONT, 0.6, 1)
    cv2.putText(canvas, tag, ((W - tw2) // 2, H - 30),
                viz.FONT, 0.6, (80, 80, 80), 1, cv2.LINE_AA)
    return canvas


# ---------------------------------------------------------------------------
# Section drawers
# ---------------------------------------------------------------------------

def _draw_header(canvas: np.ndarray, state: StudioUIState, W: int, t: float) -> None:
    """Top HUD: phase, BPM, bar, score, combo, rating flash."""
    H_hdr = _HDR_H
    # Background strip
    cv2.rectangle(canvas, (0, 0), (W, H_hdr), (28, 28, 38), -1)
    cv2.line(canvas, (0, H_hdr - 1), (W, H_hdr - 1), (55, 55, 75), 1)

    # Phase label
    phase_label = state.phase.replace("_", " ").upper()
    cv2.putText(canvas, phase_label, (16, 44), viz.FONT, 0.85, (120, 200, 255), 2, cv2.LINE_AA)

    # BPM
    bpm_str = f"{state.bpm:.0f} BPM"
    cv2.putText(canvas, bpm_str, (260, 44), viz.FONT, 0.75, (200, 200, 200), 2, cv2.LINE_AA)

    # Bar indicator
    bar_str = f"BAR {state.bar + 1}"
    cv2.putText(canvas, bar_str, (430, 44), viz.FONT, 0.75, (160, 160, 160), 1, cv2.LINE_AA)

    # Score (centered-ish)
    score_str = f"{state.score:07d}"
    (sw, _), _ = cv2.getTextSize(score_str, viz.FONT, 1.3, 3)
    sx = (W - sw) // 2
    cv2.putText(canvas, score_str, (sx, 52), viz.FONT, 1.3, (255, 240, 100), 3, cv2.LINE_AA)

    # Combo multiplier
    if state.combo > 1:
        combo_idx = min(state.combo // 5, len(_COMBO_COLORS) - 1)
        combo_col = _COMBO_COLORS[combo_idx]
        combo_str = f"x{state.combo}"
        cv2.putText(canvas, combo_str, (sx + sw + 12, 52),
                    viz.FONT, 1.0, combo_col, 2, cv2.LINE_AA)

    # Rating flash (pops up, fades)
    if state.rating:
        elapsed = t  # caller updates t continuously; use sin-based pop
        # Fade based on t oscillation — treat as a flash window
        fade = max(0.0, 1.0 - (t % 2.0))  # re-flash every 2 s while rating is set
        if fade > 0.01:
            rating_col = _RATING_COLORS.get(state.rating, (200, 200, 200))
            scale = 1.8 + 0.4 * math.sin(t * 8.0) * fade
            (rw, _), _ = cv2.getTextSize(state.rating, viz.FONT, scale, 4)
            rx = W - rw - 30
            col = tuple(int(c * fade) for c in rating_col)
            cv2.putText(canvas, state.rating, (rx, 58),
                        viz.FONT, scale, (0, 0, 0), 8, cv2.LINE_AA)
            cv2.putText(canvas, state.rating, (rx, 58),
                        viz.FONT, scale, col, 4, cv2.LINE_AA)


def _draw_center_stage(
    canvas: np.ndarray,
    state: StudioUIState,
    stage_x: int, stage_y: int,
    stage_w: int, stage_h: int,
    camera_bgr, t: float,
) -> None:
    """Center stage: camera background, live dancer skeleton, or idle attract."""
    # Camera background (dimmed)
    if camera_bgr is not None:
        cam = cv2.resize(camera_bgr, (stage_w, stage_h))
        cam_dim = (cam * 0.45).astype(np.uint8)
        canvas[stage_y:stage_y + stage_h, stage_x:stage_x + stage_w] = cam_dim

    # Stage border
    cv2.rectangle(canvas,
                  (stage_x, stage_y), (stage_x + stage_w - 1, stage_y + stage_h - 1),
                  (40, 40, 60), 1)

    if not state.present or state.live_xyz is None:
        _draw_idle_attract(canvas, stage_x, stage_y, stage_w, stage_h, t)
        return

    # Active stem color for skeleton glow
    skel_color = (160, 160, 255)
    if state.stems and 0 <= state.active_stem < len(state.stems):
        skel_color = state.stems[state.active_stem].config.color

    # Glow: draw broad thick skeleton in dim color first
    glow_col = tuple(max(0, c - 100) for c in skel_color)
    viz.draw_skeleton(canvas, state.live_xyz,
                      stage_x, stage_y, stage_w, stage_h,
                      glow_col, thick=8)
    # Main skeleton
    viz.draw_skeleton(canvas, state.live_xyz,
                      stage_x, stage_y, stage_w, stage_h,
                      skel_color, thick=3)


def _draw_idle_attract(
    canvas: np.ndarray,
    x0: int, y0: int, w: int, h: int, t: float,
) -> None:
    """Friendly idle state when no dancer is present."""
    cx, cy = x0 + w // 2, y0 + h // 2
    pulse = 0.5 + 0.5 * math.sin(t * 2.0)

    # Animated ring
    ring_col = (int(60 + 60 * pulse), int(40 + 80 * pulse), int(80 + 140 * pulse))
    cv2.circle(canvas, (cx, cy - 40), int(100 + 20 * pulse), ring_col, 2, cv2.LINE_AA)

    # "step in to play" text
    blink = 0.5 + 0.5 * math.sin(t * 2.5)
    msg = "step in to play"
    (tw, th), _ = cv2.getTextSize(msg, viz.FONT, 1.0, 2)
    col = tuple(int(c * blink) for c in (90, 240, 200))
    cv2.putText(canvas, msg, (cx - tw // 2, cy + 30),
                viz.FONT, 1.0, col, 2, cv2.LINE_AA)

    _draw_idle_figure(canvas, cx, cy - 30, t)


def _draw_idle_figure(canvas: np.ndarray, cx: int, cy: int, t: float) -> None:
    """Draw a simple animated stick-figure hint."""
    arm_sway = int(20 * math.sin(t * 1.5))
    head_y = cy - 60
    cv2.circle(canvas, (cx, head_y), 18, (70, 70, 100), 2, cv2.LINE_AA)
    # Torso
    cv2.line(canvas, (cx, head_y + 18), (cx, cy + 30), (60, 60, 90), 2, cv2.LINE_AA)
    # Arms
    cv2.line(canvas, (cx, cy - 10), (cx - 40 + arm_sway, cy + arm_sway),
             (60, 60, 90), 2, cv2.LINE_AA)
    cv2.line(canvas, (cx, cy - 10), (cx + 40 - arm_sway, cy - arm_sway),
             (60, 60, 90), 2, cv2.LINE_AA)
    # Legs
    cv2.line(canvas, (cx, cy + 30), (cx - 20, cy + 80), (60, 60, 90), 2, cv2.LINE_AA)
    cv2.line(canvas, (cx, cy + 30), (cx + 20, cy + 80), (60, 60, 90), 2, cv2.LINE_AA)


def _draw_prompt_lane(
    canvas: np.ndarray,
    state: StudioUIState,
    lane_x: int, lane_y: int, lane_w: int, lane_h: int,
    t: float,
) -> None:
    """Just-Dance move-prompt cards scrolling from right to the hit line."""
    # Lane background
    cv2.rectangle(canvas, (lane_x, lane_y), (lane_x + lane_w, lane_y + lane_h),
                  (24, 24, 34), -1)
    cv2.line(canvas, (lane_x, lane_y), (lane_x + lane_w, lane_y), (50, 50, 70), 1)
    cv2.line(canvas, (lane_x, lane_y + lane_h - 1),
             (lane_x + lane_w, lane_y + lane_h - 1), (50, 50, 70), 1)

    # Hit line
    hit_x = lane_x + int(lane_w * _HIT_LINE_X_FRAC)
    cv2.line(canvas, (hit_x, lane_y + 4), (hit_x, lane_y + lane_h - 4),
             (200, 200, 80), 2)
    cv2.putText(canvas, "HIT", (hit_x - 14, lane_y + lane_h - 6),
                viz.FONT, 0.4, (200, 200, 80), 1, cv2.LINE_AA)

    card_h = lane_h - 16
    card_w = 120

    for prompt in state.prompts:
        # Position: 0 = just appeared (right), 1 = at hit line
        frac = 1.0 - (prompt.due_t - t) / max(prompt.lead_s, 0.001)
        frac = max(-0.1, min(1.4, frac))
        # Map frac 0..1 → hit_x..lane_x+lane_w
        card_cx = int(lane_x + lane_w + (hit_x - lane_x - lane_w) * frac)
        card_x = card_cx - card_w // 2
        card_y = lane_y + 8

        move_col = _MOVE_COLORS.get(prompt.name, _MOVE_COLORS["default"])

        # Tint for hit/miss
        if prompt.hit is True:
            bg_col = (20, 80, 30)
            border_col = (80, 255, 100)
        elif prompt.hit is False:
            bg_col = (50, 20, 20)
            border_col = (80, 60, 220)
        else:
            bg_col = (30, 30, 45)
            border_col = move_col

        # Only draw if partially on screen
        if card_x + card_w < lane_x or card_x > lane_x + lane_w:
            continue

        cv2.rectangle(canvas, (card_x, card_y), (card_x + card_w, card_y + card_h),
                      bg_col, -1)
        cv2.rectangle(canvas, (card_x, card_y), (card_x + card_w, card_y + card_h),
                      border_col, 2)

        # Move name text
        label = prompt.name[:10]
        (tw, th), _ = cv2.getTextSize(label, viz.FONT, 0.5, 1)
        tx = card_x + (card_w - tw) // 2
        cv2.putText(canvas, label, (tx, card_y + card_h // 2 + th // 2),
                    viz.FONT, 0.5, move_col, 1, cv2.LINE_AA)

        # Hit/miss indicator
        if prompt.hit is True:
            cv2.putText(canvas, "OK", (card_x + card_w // 2 - 10, card_y + card_h - 6),
                        viz.FONT, 0.45, (80, 255, 100), 1, cv2.LINE_AA)
        elif prompt.hit is False:
            cv2.putText(canvas, "X", (card_x + card_w // 2 - 6, card_y + card_h - 6),
                        viz.FONT, 0.55, (80, 60, 220), 2, cv2.LINE_AA)


def _draw_stem_rack(
    canvas: np.ndarray,
    state: StudioUIState,
    rack_x: int, rack_y: int, rack_w: int, row_h: int,
) -> None:
    """Bottom stem rack: one row per stem with beat grid, meters, flags."""
    if not state.stems:
        return

    name_w = 110        # stem name column
    flag_w = 80         # MUT/SOL/REC column
    meter_col_w = 50    # cutoff + fx meters column
    grid_x = rack_x + name_w + flag_w
    grid_w = rack_w - name_w - flag_w - meter_col_w - 8

    for i, stem in enumerate(state.stems):
        y0 = rack_y + i * row_h
        active = i == state.active_stem
        col = tuple(int(c) for c in stem.config.color)

        # Row background
        bg = (32, 32, 48) if active else (22, 22, 30)
        cv2.rectangle(canvas, (rack_x, y0), (rack_x + rack_w, y0 + row_h - 1), bg, -1)
        if active:
            cv2.rectangle(canvas, (rack_x, y0), (rack_x + rack_w, y0 + row_h - 1),
                          col, 1)

        # Stem name + instrument
        name = stem.config.name[:9]
        instr = stem.config.timbre.instrument[:8]
        cv2.putText(canvas, name.upper(), (rack_x + 6, y0 + 20),
                    viz.FONT, 0.55, col, 2 if active else 1, cv2.LINE_AA)
        cv2.putText(canvas, instr, (rack_x + 6, y0 + 38),
                    viz.FONT, 0.38, (140, 140, 140), 1, cv2.LINE_AA)

        # MUTE / SOLO / REC flags
        fx0 = rack_x + name_w
        if stem.muted:
            cv2.putText(canvas, "MUT", (fx0, y0 + 20),
                        viz.FONT, 0.45, (80, 80, 220), 1, cv2.LINE_AA)
        if stem.solo:
            cv2.putText(canvas, "SOL", (fx0, y0 + 36),
                        viz.FONT, 0.45, (80, 220, 255), 1, cv2.LINE_AA)
        if stem.recording:
            # Blinking red dot
            cv2.circle(canvas, (fx0 + 60, y0 + 16), 7, (30, 30, 200), -1)
            cv2.putText(canvas, "REC", (fx0 + 38, y0 + 36),
                        viz.FONT, 0.4, (60, 60, 220), 1, cv2.LINE_AA)

        # Beat grid
        n_steps = len(stem.loop)
        if n_steps > 0:
            cell_w = max(3, grid_w // n_steps)
            cell_h = row_h - 12
            for s in range(n_steps):
                sx = grid_x + s * cell_w
                sy = y0 + 6
                has_note = stem.loop[s] is not None
                is_head = (s == state.playhead % n_steps)

                if has_note:
                    fill_col = col
                    cv2.rectangle(canvas, (sx, sy), (sx + cell_w - 1, sy + cell_h),
                                  fill_col, -1)
                else:
                    # Empty cell — dim outline
                    cv2.rectangle(canvas, (sx, sy), (sx + cell_w - 1, sy + cell_h),
                                  (45, 45, 55), 1)

                # Playhead overlay
                if is_head:
                    cv2.rectangle(canvas, (sx, sy), (sx + cell_w - 1, sy + cell_h),
                                  (255, 255, 255), 1)

        # Tiny meters: timbre cutoff + fx send (delay or reverb)
        mx = rack_x + rack_w - meter_col_w + 4
        _draw_mini_meter(canvas, stem.config.timbre.cutoff,
                         mx, y0 + 4, 16, row_h - 8, col, "CUT")
        fx_send = max(stem.config.fx.delay_send, stem.config.fx.reverb_send)
        _draw_mini_meter(canvas, fx_send,
                         mx + 26, y0 + 4, 16, row_h - 8, (180, 120, 255), "FX")


def _draw_side_meters(
    canvas: np.ndarray,
    state: StudioUIState,
    mx: int, my: int, mh: int,
) -> None:
    """Right-side vertical bars for state.meters dict."""
    if not state.meters:
        return

    items = list(state.meters.items())
    bar_h = mh - 20
    bar_w = 14
    spacing = (_METER_W - 4) // max(len(items), 1)

    for i, (name, val01) in enumerate(items):
        bx = mx + 4 + i * spacing
        by = my + 10
        val01 = max(0.0, min(1.0, float(val01)))

        # Background track
        cv2.rectangle(canvas, (bx, by), (bx + bar_w, by + bar_h),
                      (35, 35, 45), -1)
        # Filled portion
        fill_h = int(bar_h * val01)
        if fill_h > 0:
            intensity = int(80 + 175 * val01)
            bar_col = (50, intensity, intensity // 2)
            cv2.rectangle(canvas, (bx, by + bar_h - fill_h),
                          (bx + bar_w, by + bar_h), bar_col, -1)

        # Label (rotated text is complex in cv2; use abbreviated vertical label)
        label = name[:4].upper()
        cv2.putText(canvas, label, (bx - 2, by + bar_h + 14),
                    viz.FONT, 0.32, (150, 150, 150), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _draw_mini_meter(
    canvas: np.ndarray,
    val01: float, x: int, y: int, w: int, h: int,
    color: tuple, label: str,
) -> None:
    """Small vertical bar meter with a short text label below."""
    val01 = max(0.0, min(1.0, float(val01)))
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (35, 35, 45), -1)
    fill_h = int(h * val01)
    if fill_h > 0:
        cv2.rectangle(canvas, (x, y + h - fill_h), (x + w, y + h), color, -1)
    cv2.putText(canvas, label, (x - 1, y + h + 10),
                viz.FONT, 0.3, (120, 120, 120), 1, cv2.LINE_AA)
