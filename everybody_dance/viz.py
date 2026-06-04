"""Shared drawing for the song-builder UI -- used by the offline renderer
(tools/render_build.py) and the live sandbox (tools/live.py).

Imports cv2 at module load, so only import this when the real-time stack is
installed (it is not needed by the headless core / CI).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .pose import JOINT_INDEX

BONES = [("nose", "l_shoulder"), ("nose", "r_shoulder"), ("l_shoulder", "r_shoulder"),
         ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
         ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
         ("l_shoulder", "l_hip"), ("r_shoulder", "r_hip"), ("l_hip", "r_hip"),
         ("l_hip", "l_knee"), ("l_knee", "l_ankle"),
         ("r_hip", "r_knee"), ("r_knee", "r_ankle")]
COLORS = {"drums": (80, 120, 255), "bass": (80, 220, 120),
          "keys": (240, 180, 70), "lead": (200, 100, 240)}   # BGR
BADGE = {"pending": "", "rhythm": "REC RHYTHM", "pitch": "REC PITCH", "saved": "loop"}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _proj(xyz, x0, y0, w, h):
    pts = {}
    for j, i in JOINT_INDEX.items():
        x = (xyz[i, 0] + 0.8) / 1.6
        y = 1.0 - (xyz[i, 1] + 1.1) / 2.0
        pts[j] = (int(x0 + x * w), int(y0 + y * h))
    return pts


def draw_skeleton(img, xyz, x0, y0, w, h, color, thick=2):
    p = _proj(xyz, x0, y0, w, h)
    for a, b in BONES:
        cv2.line(img, p[a], p[b], color, thick, cv2.LINE_AA)
    for j in p:
        cv2.circle(img, p[j], 3, color, -1, cv2.LINE_AA)


def draw_skeleton_px(img, px: Optional[np.ndarray], color=(255, 255, 255), thick=2):
    """Overlay a skeleton on a camera image from pixel landmarks (13x2) or None."""
    if px is None:
        return
    p = {j: (int(px[i, 0]), int(px[i, 1])) for j, i in JOINT_INDEX.items()}
    for a, b in BONES:
        cv2.line(img, p[a], p[b], color, thick, cv2.LINE_AA)
    for j in p:
        cv2.circle(img, p[j], 4, color, -1, cv2.LINE_AA)


def draw_instrument_panel(canvas, role, ui, x0, y0, w, h):
    col = COLORS[role]
    status = ui.status[role]
    active = role == ui.active
    cv2.rectangle(canvas, (x0 + 2, y0 + 2), (x0 + w - 2, y0 + h - 2),
                  col if active else (55, 55, 55), 2 if active else 1)
    cv2.putText(canvas, role.upper(), (x0 + 10, y0 + 22), FONT, 0.55,
                col if status != "pending" else (90, 90, 90), 2)
    if active:
        draw_skeleton(canvas, ui.live_xyz, x0, y0 + 24, w, h - 50, (255, 255, 255), 2)
    elif status == "saved" and ui.ghosts.get(role) is not None:
        draw_skeleton(canvas, ui.ghosts[role], x0, y0 + 24, w, h - 50, col, 2)
    if BADGE[status]:
        cv2.putText(canvas, BADGE[status], (x0 + 10, y0 + h - 10), FONT, 0.5,
                    (255, 255, 255) if active else col, 1)


def draw_timeline(canvas, ui, loop_steps, x0, y0, w, row_h=22):
    for k, role in enumerate(ui.order):
        y = y0 + k * row_h
        cv2.putText(canvas, role[:2], (x0, y + 14), FONT, 0.4, COLORS[role], 1)
        n = len(ui.onsets[role])
        for s, present in enumerate(ui.onsets[role]):
            if present:
                sx = x0 + 26 + int(s / n * (w - 26))
                cv2.rectangle(canvas, (sx, y), (sx + 4, y + row_h - 6), COLORS[role], -1)
    px = x0 + 26 + int(ui.playhead / loop_steps * (w - 26))
    cv2.line(canvas, (px, y0 - 4), (px, y0 + len(ui.order) * row_h), (255, 255, 255), 1)


def draw_pitch_meter(canvas, pitch01, x0, y0, h, w=12):
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (60, 60, 60), 1)
    my = int(y0 + h - pitch01 * h)
    cv2.rectangle(canvas, (x0, my), (x0 + w, y0 + h), (200, 200, 80), -1)
    cv2.putText(canvas, "pitch", (x0 - 4, y0 + h + 16), FONT, 0.4, (200, 200, 80), 1)


# -- whole-frame compositions ---------------------------------------------

def compose_offline(ui, W, H, loop_steps):
    """The 4-panels-in-a-row layout used by render_build.py."""
    img = np.full((H, W, 3), 24, np.uint8)
    panel_h, tl_y = H - 160, H - 130
    pw = W // len(ui.order)
    for k, role in enumerate(ui.order):
        draw_instrument_panel(img, role, ui, k * pw, 28, pw, panel_h - 28)
    ph = ui.phase.replace("_", " ").upper()
    cv2.putText(img, f"{ph}   {ui.bars_left} bars   |   {ui.bpm:.0f} BPM",
                (16, H - 138), FONT, 0.55, (220, 220, 220), 1)
    draw_pitch_meter(img, ui.pitch01, W - 22, 28, panel_h - 28)
    draw_timeline(img, ui, loop_steps, 30, tl_y, W - 60, 26)
    return img


def compose_live(camera_bgr, ui, perf: Dict, loop_steps, W=1280, H=720):
    canvas = np.full((H, W, 3), 22, np.uint8)
    # --- header / HUD ---
    cv2.putText(canvas, "everybodyDance  -  dance continuously, the song builds",
                (16, 26), FONT, 0.7, (235, 235, 235), 2)
    ph = ui.phase.replace("_", " ").upper()
    cv2.putText(canvas, f"{ph}   {ui.bars_left} bars left    {ui.bpm:.0f} BPM",
                (16, 54), FONT, 0.6, (120, 220, 255), 2)
    hud = (f"cam {perf.get('fps', 0):.0f}fps | infer {perf.get('infer_ms', 0):.0f}ms"
           f" | draw {perf.get('draw_ms', 0):.0f}ms | audio "
           f"{'ON' if perf.get('audio') else 'off'}")
    cv2.putText(canvas, hud, (16, 78), FONT, 0.5, (170, 170, 170), 1)
    if perf.get("note"):
        cv2.putText(canvas, perf["note"], (16, H - 12), FONT, 0.5, (160, 160, 160), 1)

    # --- camera with skeleton overlay (top-left) ---
    cx, cy, cw, ch = 16, 92, 540, 405
    if camera_bgr is not None:
        cam = cv2.resize(camera_bgr, (cw, ch))
        draw_skeleton_px(cam, perf.get("landmarks_px"))
        canvas[cy:cy + ch, cx:cx + cw] = cam
    cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), (70, 70, 70), 1)

    # --- 2x2 instrument grid (right of camera) ---
    gx, gy, gw, gh, gap = 580, 92, 690, 405, 8
    cellw, cellh = (gw - gap) // 2, (gh - gap) // 2
    for k, role in enumerate(ui.order):
        x0 = gx + (k % 2) * (cellw + gap)
        y0 = gy + (k // 2) * (cellh + gap)
        draw_instrument_panel(canvas, role, ui, x0, y0, cellw, cellh)

    # --- pitch meter + timeline (bottom) ---
    draw_pitch_meter(canvas, ui.pitch01, W - 30, 92, 405)
    draw_timeline(canvas, ui, loop_steps, 30, H - 200, W - 80, 26)
    return canvas
