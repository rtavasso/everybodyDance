"""The gallery Just-Dance visual identity -- the screen a visitor sees.

A pure rendering layer over :class:`studio.StudioUI`: it draws the dancer as a
glowing, mood-coloured silhouette with a motion trail, the five-stem "band" as
lanes (VU + loop grid + REC/LOOP/MUTE badges), the recognised "gold move" cards,
the onboarding prompt, and a compact HUD -- on a coherent dark gallery canvas
legible from across a room. The empty-room invitation lives in
:func:`render_attract`.

cv2 is imported at module load (same contract as :mod:`viz`), so import this only
where the real-time/drawing stack is present -- it is not needed by the headless
core. Everything draws deterministically from numpy + cv2; the only "state" is a
short skeleton trail held by :class:`Stage`, and the thin module-level
``compose_stage``/``render_attract`` wrappers are statelessly callable (they spin
up a private shared instance) so the offline grader can call them as plain
functions on a None camera + zero skeleton.
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from . import viz
from .effects import STYLE
from .pose import JOINT_INDEX, JOINTS

FONT = viz.FONT
BG = (16, 14, 20)                         # the gallery dark
PANEL = (30, 26, 36)

# Skeleton fit for the centre stage (the live_xyz is hip-centred, torso=1).
XR, YR = (-1.5, 1.5), (-2.3, 1.6)

# Mood (scale name) -> a body accent BGR. Warm/bright for major-ish, cool/dark
# for minor/phrygian, so the dancer changes hue as the music's mood shifts.
MOOD_COLOR = {
    "major": (120, 230, 255), "major_pentatonic": (140, 240, 255),
    "lydian": (200, 255, 200), "dorian": (255, 210, 120),
    "mixolydian": (160, 230, 255),
    "minor": (255, 170, 120), "minor_pentatonic": (255, 160, 150),
    "phrygian": (220, 130, 255),
}
DEFAULT_MOOD_COLOR = (220, 220, 235)

GOLD = (60, 215, 255)                     # gold-move / GOLD-tier accent (BGR)
TIER_COLOR = {"": (120, 120, 130), "WARM": (120, 200, 255),
              "FIRE": (80, 130, 255), "GOLD": GOLD}

# Song-arc section -> the aurora's base tint (BGR, dusk-dark; heat scales it).
SECTION_TINT = {"intro": (58, 28, 22), "groove": (52, 44, 18),
                "build": (26, 40, 66), "peak": (66, 26, 70)}


def mood_color(mood: str) -> tuple:
    return MOOD_COLOR.get(mood, DEFAULT_MOOD_COLOR)


def _scaled(color, k):
    return tuple(int(np.clip(c * k, 0, 255)) for c in color)


# -- skeleton projection (mirrors viz._proj but returns the array) ----------

def _project(xyz, x0, y0, w, h, xr=XR, yr=YR):
    pts = np.zeros((len(JOINTS), 2), np.int32)
    for j, i in JOINT_INDEX.items():
        x = (xyz[i, 0] - xr[0]) / (xr[1] - xr[0])
        y = 1.0 - (xyz[i, 1] - yr[0]) / (yr[1] - yr[0])
        pts[i] = (int(x0 + x * w), int(y0 + y * h))
    return pts


def _glow_skeleton(img, pts, color, thick=4, glow=True):
    """A skeleton drawn as bones+joints; when `glow`, a soft wide pass under it."""
    if glow:
        ov = img.copy()
        for a, b in viz.BONES:
            cv2.line(ov, tuple(pts[JOINT_INDEX[a]]), tuple(pts[JOINT_INDEX[b]]),
                     _scaled(color, 0.85), thick + 10, cv2.LINE_AA)
        cv2.addWeighted(ov, 0.35, img, 0.65, 0, img)
    for a, b in viz.BONES:
        cv2.line(img, tuple(pts[JOINT_INDEX[a]]), tuple(pts[JOINT_INDEX[b]]),
                 color, thick, cv2.LINE_AA)
    for i in range(len(pts)):
        cv2.circle(img, tuple(pts[i]), thick + 1, color, -1, cv2.LINE_AA)


def _vignette(img):
    """A radial darkening so the bright centre stage reads against the room."""
    H, W = img.shape[:2]
    ys, xs = np.mgrid[0:H, 0:W]
    cx, cy = W * 0.5, H * 0.52
    r = np.sqrt(((xs - cx) / (W * 0.62)) ** 2 + ((ys - cy) / (H * 0.62)) ** 2)
    m = np.clip(1.0 - 0.55 * r, 0.45, 1.0).astype(np.float32)
    img[:] = (img.astype(np.float32) * m[..., None]).astype(np.uint8)


def _has_skeleton(xyz) -> bool:
    return xyz is not None and float(np.abs(xyz).sum()) > 1e-6


class Stage:
    """Holds the motion trail (and is the home for `compose_stage`/`render_attract`).

    The trail is a short ring of past projected skeletons that fade out; a beat
    pulse and a handful of beat-synced particles ride on top. Keep one instance
    per live window (so the trail is per-screen); the headless grader can also
    just call the module-level wrappers, which own a private shared instance.
    """

    TRAIL = 8

    def __init__(self, W: int = 1280, H: int = 720):
        self.W, self.H = W, H
        self._trail: list = []            # list[(pts(J,2) int, age 0..1)]
        self._pulse = 0.0                 # beat pulse envelope, decays per frame
        self._last_beat = False

    # -- centre stage ------------------------------------------------------

    def _aurora(self, canvas, ui):
        """Slow rolling colour bands behind everything: tinted by the song-arc
        section, blended with the mood, brightened by heat. Pure function of
        (ui, t) -- deterministic, and the whole room changes colour as the
        song travels intro -> peak."""
        g = getattr(ui, "game", None)
        H = canvas.shape[0]
        t = float(ui.t)
        mood = np.array(mood_color(ui.mood), np.float32)
        sec = np.array(SECTION_TINT.get(g.section if g else "groove",
                                        (40, 36, 40)), np.float32)
        heat = float(g.heat) if g else 0.3
        ys = np.linspace(0.0, 1.0, H, dtype=np.float32)
        band = 0.5 + 0.5 * np.sin(2 * np.pi * (ys * 1.5 - 0.10 * t))
        band2 = 0.5 + 0.5 * np.sin(2 * np.pi * (ys * 2.3 + 0.06 * t) + 1.7)
        glow = (0.07 + 0.13 * heat) * band + (0.04 + 0.08 * heat) * band2
        col = 0.65 * sec + 0.35 * mood
        add = glow[:, None, None] * col[None, None, :]
        canvas[:] = np.clip(canvas.astype(np.float32) + add, 0,
                            255).astype(np.uint8)

    def _shockwaves(self, canvas, ui, cx, cy, w, h):
        """Expanding rings from the dancer on every big flash (combos, drops,
        PERFECT) -- the musical slam gets a physical blast wave."""
        for f in ui.flashes:
            if not getattr(f, "big", False):
                continue
            a = f.alpha(ui.t)
            if a <= 0:
                continue
            prog = 1.0 - a
            ov = canvas.copy()
            for k in range(3):
                r = int((prog * 1.1 + 0.14 * k) * min(w, h) * 0.62) + 8
                cv2.circle(ov, (cx, cy), r, f.color, max(2, int(5 * a)),
                           cv2.LINE_AA)
            cv2.addWeighted(ov, 0.5 * a, canvas, 1 - 0.5 * a, 0, canvas)

    def _confetti(self, canvas, ui):
        """A deterministic golden confetti burst while a PERFECT! flash lives:
        golden-angle scatter, per-particle fall speed and sway from k alone."""
        W = int(self.W * 0.62)
        for f in ui.flashes:
            if f.name != "PERFECT!":
                continue
            age = float(np.clip((ui.t - f.t0) / max(f.ttl, 1e-6), 0.0, 1.0))
            for k in range(40):
                phi = k * 2.399963                       # golden angle
                x = int(W * 0.5 + 0.42 * W * np.cos(phi)
                        * (0.25 + 0.75 * ((k * 37) % 100) / 100.0))
                speed = 180 + 3 * ((k * 53) % 100)
                y = int(self.H * 0.22 + age * speed
                        + 26 * np.sin(phi * 3 + age * 7))
                col = (GOLD, (255, 255, 255),
                       mood_color(ui.mood))[k % 3]
                if 0 <= x < self.W and 0 <= y < self.H:
                    cv2.circle(canvas, (x, y), 2 + (k % 2), col, -1,
                               cv2.LINE_AA)

    def _draw_body(self, canvas, ui, x0, y0, w, h):
        col = mood_color(ui.mood)
        g = getattr(ui, "game", None)
        if g is not None and g.streak_tier == "GOLD":
            # a GOLD streak gilds the dancer.
            col = tuple(int(0.45 * c + 0.55 * gc) for c, gc in zip(col, GOLD))
        has = _has_skeleton(ui.live_xyz)

        # beat pulse: a ring expanding from the body on each beat edge.
        if ui.beat and not self._last_beat:
            self._pulse = 1.0
        self._last_beat = bool(ui.beat)

        cx, cy = x0 + w // 2, y0 + h // 2
        if self._pulse > 0.02:
            rad = int((1.0 - self._pulse) * min(w, h) * 0.55 + 30)
            ov = canvas.copy()
            cv2.circle(ov, (cx, cy), rad, _scaled(col, 0.9),
                       max(2, int(8 * self._pulse)), cv2.LINE_AA)
            cv2.addWeighted(ov, 0.5 * self._pulse, canvas, 1 - 0.5 * self._pulse,
                            0, canvas)
        self._pulse *= 0.82
        self._shockwaves(canvas, ui, cx, cy, w, h)

        if not has:
            self._trail.clear()
            cv2.putText(canvas, "step in to play", (cx - 150, cy),
                        FONT, 1.1, _scaled(col, 0.8), 2, cv2.LINE_AA)
            return

        pts = _project(ui.live_xyz, x0, y0, w, h)
        # motion trail: fade older skeletons behind the live one.
        self._trail.append(pts.copy())
        if len(self._trail) > self.TRAIL:
            self._trail.pop(0)
        n = len(self._trail)
        for k, past in enumerate(self._trail[:-1]):
            a = (k + 1) / n
            ov = canvas.copy()
            tcol = _scaled(col, 0.25 + 0.4 * a)
            for bo in viz.BONES:
                cv2.line(ov, tuple(past[JOINT_INDEX[bo[0]]]),
                         tuple(past[JOINT_INDEX[bo[1]]]), tcol,
                         max(1, int(2 + 3 * a)), cv2.LINE_AA)
            cv2.addWeighted(ov, 0.35 * a, canvas, 1 - 0.35 * a, 0, canvas)

        thick = 4 + int(4 * float(np.clip(ui.energy, 0, 1)))
        _glow_skeleton(canvas, pts, col, thick=thick, glow=True)

        # beat-synced particles: shoot from the hands on a beat, energy-scaled.
        if (ui.beat or self._pulse > 0.4) and ui.energy > 0.04:
            count = 2 + int(8 * float(np.clip(ui.energy, 0, 1)))
            for w_name in ("l_wrist", "r_wrist"):
                hand = pts[JOINT_INDEX[w_name]]
                for p in range(count):
                    ang = (p / max(count, 1)) * 2 * math.pi
                    rr = int(14 + 26 * self._pulse + 10 * (p % 3))
                    px = int(hand[0] + rr * math.cos(ang))
                    py = int(hand[1] + rr * math.sin(ang))
                    cv2.circle(canvas, (px, py), 2, _scaled(col, 0.9), -1,
                               cv2.LINE_AA)

    # -- the band: one lane per stem --------------------------------------

    def _draw_stems(self, canvas, ui, x0, y0, w, lane_h, gap):
        for k, s in enumerate(ui.stems):
            y = y0 + k * (lane_h + gap)
            self._draw_lane(canvas, s, ui, x0, y, w, lane_h)

    def _draw_lane(self, canvas, s, ui, x0, y0, w, h):
        col = tuple(int(c) for c in s.color)
        if getattr(s, "locked", False):
            # a stem the song arc hasn't unlocked yet: dim, with the invitation.
            cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), PANEL, -1)
            cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (52, 48, 60), 1)
            cv2.putText(canvas, s.name.upper(), (x0 + 12, y0 + 24), FONT, 0.62,
                        _scaled(col, 0.35), 2, cv2.LINE_AA)
            cv2.putText(canvas, "LOCKED - keep dancing", (x0 + 12, y0 + h // 2 + 12),
                        FONT, 0.5, (110, 106, 120), 1, cv2.LINE_AA)
            return
        active = s.active or s.level > 0.05
        cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), PANEL, -1)
        edge = col if active else _scaled(col, 0.4)
        cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), edge,
                      2 if active else 1)

        # name + timbre
        cv2.putText(canvas, s.name.upper(), (x0 + 12, y0 + 24), FONT, 0.62,
                    col, 2, cv2.LINE_AA)
        cv2.putText(canvas, s.timbre, (x0 + 12, y0 + 44), FONT, 0.42,
                    _scaled(col, 0.7), 1, cv2.LINE_AA)

        # VU / level bar (left third)
        bx, bw = x0 + 12, 90
        by, bh = y0 + 56, h - 70
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (50, 46, 56), 1)
        fill = int(bw * float(np.clip(s.level, 0, 1)))
        cv2.rectangle(canvas, (bx, by), (bx + fill, by + bh), col, -1)
        cv2.putText(canvas, "VU", (bx, by + bh + 14), FONT, 0.36,
                    (140, 140, 150), 1, cv2.LINE_AA)
        # note sparks: tiny deterministic glints while the lane is sounding.
        if active and s.level > 0.1:
            for k in range(4):
                sx = x0 + 14 + ((int(ui.t * 90) * (k + 3) + k * 41)
                                % max(w - 28, 1))
                sy = y0 + 8 + ((k * 29 + int(ui.t * 60)) % 12)
                cv2.circle(canvas, (sx, sy), 1 + (k % 2), _scaled(col, 0.9),
                           -1, cv2.LINE_AA)

        # loop grid: onsets across the loop, playhead highlighted.
        gx0 = bx + bw + 24
        gw = x0 + w - gx0 - 12
        gy = y0 + 60
        gh = h - 78
        onsets = s.onsets or []
        nsteps = len(onsets)
        cv2.rectangle(canvas, (gx0, gy), (gx0 + gw, gy + gh), (44, 40, 50), 1)
        if nsteps:
            cell = gw / nsteps
            for i, on in enumerate(onsets):
                cxa = int(gx0 + i * cell)
                cxb = int(gx0 + (i + 1) * cell) - 1
                if i == ui.playhead:
                    cv2.rectangle(canvas, (cxa, gy), (cxb, gy + gh),
                                  _scaled(col, 0.55), -1)
                if on:
                    cv2.rectangle(canvas, (cxa + 1, gy + 2), (cxb - 1, gy + gh - 2),
                                  col, -1)
                    cv2.rectangle(canvas, (cxa + 1, gy + 2), (cxb - 1, gy + gh - 2),
                                  _scaled(col, 1.2), 1)
        # playhead marker line on top of the grid
        if nsteps:
            px = int(gx0 + (ui.playhead + 0.5) / nsteps * gw)
            cv2.line(canvas, (px, gy - 4), (px, gy + gh + 4), (235, 235, 245), 1,
                     cv2.LINE_AA)

        # status badges (REC/LOOP/MUTE) top-right of the lane
        badges = []
        if s.recording:
            badges.append(("REC", (80, 80, 255)))
        if s.looping:
            badges.append(("LOOP", (120, 255, 160)))
        if s.muted:
            badges.append(("MUTE", (120, 120, 130)))
        bx2 = x0 + w - 12
        for label, bcol in reversed(badges):
            (tw, _), _ = cv2.getTextSize(label, FONT, 0.42, 1)
            bx2 -= tw + 16
            cv2.rectangle(canvas, (bx2, y0 + 10), (bx2 + tw + 10, y0 + 30),
                          bcol, -1)
            cv2.putText(canvas, label, (bx2 + 5, y0 + 25), FONT, 0.42,
                        (20, 18, 24), 1, cv2.LINE_AA)

    # -- HUD ---------------------------------------------------------------

    def _draw_hud(self, canvas, ui, perf):
        W, H = self.W, self.H
        cv2.putText(canvas, "everybodyDance", (24, 36), FONT, 0.9,
                    (235, 235, 245), 2, cv2.LINE_AA)
        line = (f"{ui.phase.upper()}   {ui.bpm:4.0f} BPM   "
                f"mood:{ui.mood}")
        cv2.putText(canvas, line, (24, 62), FONT, 0.6, mood_color(ui.mood),
                    2, cv2.LINE_AA)
        hud = (f"cam {perf.get('fps', 0):.0f}fps | infer "
               f"{perf.get('infer_ms', 0):.0f}ms | audio "
               f"{'ON' if perf.get('audio') else 'off'}")
        cv2.putText(canvas, hud, (24, H - 16), FONT, 0.48, (150, 150, 160), 1,
                    cv2.LINE_AA)
        if perf.get("note"):
            (tw, _), _ = cv2.getTextSize(perf["note"], FONT, 0.48, 1)
            cv2.putText(canvas, perf["note"], (W - tw - 24, H - 16), FONT, 0.48,
                        (150, 150, 160), 1, cv2.LINE_AA)

    # -- game layer: arc / streak / gold-move / identity --------------------

    def _draw_game(self, canvas, ui):
        """Section tracker + streak meter, top-centre of the stage half."""
        g = getattr(ui, "game", None)
        if g is None:
            return
        cx = int(self.W * 0.62) // 2 + 60        # clear of the title block
        labels = [s.upper() for s in g.sections]
        widths = [cv2.getTextSize(l, FONT, 0.48, 1)[0][0] for l in labels]
        total = sum(widths) + 22 * (len(labels) - 1)
        x = cx - total // 2
        for label, lw, name in zip(labels, widths, g.sections):
            cur = (name == g.section)
            col = mood_color(ui.mood) if cur else (104, 100, 114)
            cv2.putText(canvas, label, (x, 36), FONT, 0.48, col,
                        2 if cur else 1, cv2.LINE_AA)
            if cur:
                cv2.line(canvas, (x, 42), (x + lw, 42), col, 2, cv2.LINE_AA)
            x += lw + 22

        bw, bh = 220, 7
        bx, by = cx - bw // 2, 50
        tcol = TIER_COLOR.get(g.streak_tier, TIER_COLOR[""])
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (50, 46, 56), 1)
        fill = int(bw * float(np.clip(g.streak, 0, 1)))
        if fill > 0:
            cv2.rectangle(canvas, (bx, by), (bx + fill, by + bh), tcol, -1)
        if g.streak_tier:
            cv2.putText(canvas, g.streak_tier, (bx + bw + 10, by + bh + 1),
                        FONT, 0.45, tcol, 1, cv2.LINE_AA)

    def _draw_challenge(self, canvas, ui):
        """The gold-move card: announce ('GET READY'), then the timed window."""
        g = getattr(ui, "game", None)
        if (g is None or not g.challenge_move
                or g.challenge_state not in ("announce", "window")):
            return
        w, h = 240, 66
        x1 = int(self.W * 0.62) - 40
        x0, y0 = x1 - w, 86
        live = g.challenge_state == "window"
        cv2.rectangle(canvas, (x0, y0), (x1, y0 + h), (30, 30, 24), -1)
        cv2.rectangle(canvas, (x0, y0), (x1, y0 + h), GOLD, 3 if live else 1)
        head = "* GOLD MOVE -- NOW!" if live else "* GOLD MOVE  get ready"
        cv2.putText(canvas, head, (x0 + 10, y0 + 20), FONT, 0.42,
                    GOLD if live else (180, 180, 190), 1, cv2.LINE_AA)
        cv2.putText(canvas, g.challenge_move, (x0 + 10, y0 + 48), FONT, 0.85,
                    GOLD, 2, cv2.LINE_AA)
        # time draining out of the phase, right-to-left
        rem = int(w * float(np.clip(1.0 - g.challenge_frac, 0, 1)))
        cv2.rectangle(canvas, (x0, y0 + h - 5), (x0 + rem, y0 + h - 2), GOLD, -1)

    def _draw_identity(self, canvas, ui):
        """The Dance-DNA card: this visitor's stage name + sonic world."""
        g = getattr(ui, "game", None)
        if g is None or not g.identity_name:
            return
        x0, y1 = 24, self.H - 36
        col = tuple(int(c) for c in g.identity_color)
        cv2.putText(canvas, "YOUR SOUND", (x0, y1 - 38), FONT, 0.38,
                    (150, 150, 160), 1, cv2.LINE_AA)
        cv2.putText(canvas, g.identity_name, (x0, y1 - 16), FONT, 0.72, col,
                    2, cv2.LINE_AA)
        cv2.putText(canvas, g.identity_tagline, (x0, y1 + 2), FONT, 0.45,
                    (175, 175, 185), 1, cv2.LINE_AA)

    def _draw_prompt(self, canvas, ui):
        """Onboarding hint: surface the cycling prompt when idle/low-energy."""
        if not ui.prompt:
            return
        W = self.W
        text = ui.prompt
        (tw, th), _ = cv2.getTextSize(text, FONT, 1.0, 2)
        x = (W - tw) // 2
        y = int(self.H * 0.86)
        cv2.rectangle(canvas, (x - 24, y - th - 16), (x + tw + 24, y + 16),
                      (28, 24, 34), -1)
        cv2.rectangle(canvas, (x - 24, y - th - 16), (x + tw + 24, y + 16),
                      (90, 86, 100), 1)
        cv2.putText(canvas, "TRY:", (x - 24 + 8, y - th - 16 - 6), FONT, 0.45,
                    (160, 160, 170), 1, cv2.LINE_AA)
        cv2.putText(canvas, text, (x, y), FONT, 1.0, (245, 245, 255), 2,
                    cv2.LINE_AA)

    # -- whole-frame composition ------------------------------------------

    def compose(self, camera_bgr, ui, perf, W=None, H=None,
                draw_body=True, cam_gain=0.32):
        """``draw_body=False`` + a high ``cam_gain`` is the session-film mode:
        the real camera (with its own pose overlay) IS the dancer, so the
        abstract centre-stage skeleton steps aside; pulses/shockwaves stay."""
        W = W or self.W
        H = H or self.H
        if (W, H) != (self.W, self.H):
            self.W, self.H = W, H
        canvas = np.empty((H, W, 3), np.uint8)
        canvas[:] = BG
        self._aurora(canvas, ui)

        # left ~62%: the centre stage; right ~38%: the band lanes.
        stage_w = int(W * 0.62)
        # camera as a dim backdrop behind the dancer (if provided).
        if camera_bgr is not None:
            cam = cv2.resize(camera_bgr, (stage_w, H))
            cam = (cam.astype(np.float32) * float(cam_gain)).astype(np.uint8)
            canvas[:, :stage_w] = np.maximum(canvas[:, :stage_w], cam)

        if draw_body:
            self._draw_body(canvas, ui, 30, 80, stage_w - 60, H - 200)
        else:
            cx, cy = 30 + (stage_w - 60) // 2, 80 + (H - 200) // 2
            if ui.beat and not self._last_beat:
                self._pulse = 1.0
            self._last_beat = bool(ui.beat)
            self._pulse *= 0.82
            self._shockwaves(canvas, ui, cx, cy, stage_w - 60, H - 200)
        _vignette(canvas)

        # band lanes on the right
        bx = stage_w + 16
        bw = W - bx - 16
        n = max(1, len(ui.stems))
        top, bot = 84, 60
        gap = 10
        lane_h = (H - top - bot - gap * (n - 1)) // n
        self._draw_stems(canvas, ui, bx, top, bw, lane_h, gap)
        cv2.putText(canvas, "THE BAND", (bx, top - 14), FONT, 0.5,
                    (160, 160, 170), 1, cv2.LINE_AA)

        self._draw_hud(canvas, ui, perf)
        self._draw_game(canvas, ui)
        self._draw_challenge(canvas, ui)
        self._draw_identity(canvas, ui)
        self._draw_prompt(canvas, ui)
        self._confetti(canvas, ui)

        # gold-move cards last, on top of everything (reuse viz.draw_flashes).
        if ui.flashes:
            viz.draw_flashes(canvas, ui.flashes, ui.t)
        return canvas

    # -- attract loop ------------------------------------------------------

    def attract(self, t, W=None, H=None):
        W = W or self.W
        H = H or self.H
        canvas = np.empty((H, W, 3), np.uint8)
        canvas[:] = BG
        cx, cy = W // 2, int(H * 0.52)

        # a slow breathing field of concentric rings (deterministic from t)
        for k in range(5):
            phase = t * 0.6 - k * 0.5
            r = int((0.5 + 0.5 * math.sin(phase)) * 120 + 60 + k * 60)
            a = 0.5 * (0.5 + 0.5 * math.cos(phase))
            ov = canvas.copy()
            cv2.circle(ov, (cx, cy), r, (120, 110, 160), 2, cv2.LINE_AA)
            cv2.addWeighted(ov, a, canvas, 1 - a, 0, canvas)

        # a pulsing silhouette outline: a stylised standing figure that breathes.
        breath = 0.5 + 0.5 * math.sin(t * 1.4)
        col = _scaled((200, 180, 255), 0.5 + 0.5 * breath)
        scale = 1.0 + 0.04 * math.sin(t * 1.4)
        figure = _attract_skeleton(t)
        pts = _project(figure, cx - int(W * 0.18 * scale), int(H * 0.16),
                       int(W * 0.36 * scale), int(H * 0.64), xr=XR, yr=YR)
        _glow_skeleton(canvas, pts, col, thick=5, glow=True)
        _vignette(canvas)

        # text-light invitation that gently pulses in/out.
        alpha = 0.55 + 0.45 * breath
        title = "everybodyDance"
        (tw, _), _ = cv2.getTextSize(title, FONT, 1.4, 3)
        cv2.putText(canvas, title, ((W - tw) // 2, 80), FONT, 1.4,
                    _scaled((235, 235, 245), alpha), 3, cv2.LINE_AA)
        invite = "step in to play"
        (iw, _), _ = cv2.getTextSize(invite, FONT, 1.1, 2)
        cv2.putText(canvas, invite, ((W - iw) // 2, int(H * 0.9)), FONT, 1.1,
                    _scaled((200, 200, 230), alpha), 2, cv2.LINE_AA)
        return canvas


def _attract_skeleton(t):
    """A gently swaying standing figure (torso-normalised, like live_xyz)."""
    sway = 0.12 * math.sin(t * 1.4)
    arm = 0.5 + 0.4 * math.sin(t * 1.4)
    s = {
        "nose": (sway, 1.35, 0.0),
        "l_shoulder": (-0.25 + sway, 1.0, 0.0), "r_shoulder": (0.25 + sway, 1.0, 0.0),
        "l_elbow": (-0.45, 0.55, 0.0), "r_elbow": (0.45, 0.55, 0.0),
        "l_wrist": (-0.5, 0.1 + arm * 0.5, 0.0), "r_wrist": (0.5, 0.1 + arm * 0.5, 0.0),
        "l_hip": (-0.15, 0.0, 0.0), "r_hip": (0.15, 0.0, 0.0),
        "l_knee": (-0.17, -1.0, 0.0), "r_knee": (0.17, -1.0, 0.0),
        "l_ankle": (-0.18, -2.0, 0.0), "r_ankle": (0.18, -2.0, 0.0),
    }
    return np.array([s[j] for j in JOINTS], float)


# -- module-level wrappers (statelessly callable; own a shared Stage) -------

_SHARED: Optional[Stage] = None


def _shared(W, H) -> Stage:
    global _SHARED
    if _SHARED is None or (_SHARED.W, _SHARED.H) != (W, H):
        _SHARED = Stage(W, H)
    return _SHARED


def compose_stage(camera_bgr, ui, perf: dict, W: int = 1280, H: int = 720):
    """Render one gallery frame: dancer + band + flashes + HUD.

    Fully headless-capable: ``camera_bgr`` may be None (a generated dark stage is
    drawn instead), so the SAME function renders the live screen and the offline
    gradable frames. Robust to a zero skeleton / empty flashes / empty onsets.
    """
    return _shared(W, H).compose(camera_bgr, ui, perf or {}, W, H)


def render_attract(t: float, W: int = 1280, H: int = 720):
    """The empty-room idle/onboarding loop: a pulsing silhouette + invitation,
    animating purely (and deterministically) from ``t``."""
    return _shared(W, H).attract(t, W, H)
