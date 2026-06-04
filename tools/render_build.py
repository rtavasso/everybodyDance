#!/usr/bin/env python3
"""Render the song-builder: dance continuously, the song builds, and each
committed stem becomes a looping skeleton ghost. Outputs an overlay MP4 (the
4-panel chorus + loop timeline) and the WAV it produces.
"""

import argparse
import os

import numpy as np

from everybody_dance.builder import SongBuilder, BuilderConfig
from everybody_dance.features import FeatureExtractor
from everybody_dance.laban import LabanEstimator
from everybody_dance.oscillator import EntrainedClock
from everybody_dance.output import LogBackend
from everybody_dance.personalization import Calibrator
from everybody_dance.pose import JOINT_INDEX
from everybody_dance.sources import ArrayPoseSource
from tools.run_dataset import _load
from tools.synth import render as render_wav

BONES = [("nose", "l_shoulder"), ("nose", "r_shoulder"), ("l_shoulder", "r_shoulder"),
         ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
         ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
         ("l_shoulder", "l_hip"), ("r_shoulder", "r_hip"), ("l_hip", "r_hip"),
         ("l_hip", "l_knee"), ("l_knee", "l_ankle"),
         ("r_hip", "r_knee"), ("r_knee", "r_ankle")]
COLORS = {"drums": (80, 120, 255), "bass": (80, 220, 120),
          "keys": (240, 180, 70), "lead": (200, 100, 240)}   # BGR


def energetic_window(src, seconds):
    fe = FeatureExtractor(src.fps)
    vals = np.array([(f.core_energy_env + f.limb_energy_env) for f in
                     (fe.update(fr) for fr in src.frames())])
    W = int(seconds * src.fps)
    if len(vals) <= W:
        return src
    start = int(np.argmax(np.convolve(vals, np.ones(W) / W, "valid")))
    return ArrayPoseSource(src.xyz_seq[start:start + W], fps=src.fps,
                           flip_y=src.flip_y, name=src.name)


def calibrate(src):
    fe, lab, clk, cal = FeatureExtractor(src.fps), LabanEstimator(), EntrainedClock(), Calibrator()
    idx = [JOINT_INDEX[j] for j in ("l_wrist", "r_wrist", "l_ankle", "r_ankle")]
    hits = []
    for fr in src.frames():
        f = fe.update(fr)
        cal.observe(f, lab.update(f, f.dt), clk.update(float(f.bounce), f.dt).tempo_hz)
        hits.append(float(f.accel_mag[idx].max()))
    prof = cal.finalize(signature=fe.signature.summary())
    tempo = prof.char_tempo_hz
    while tempo < 1.0:
        tempo *= 2
    while tempo > 2.3:
        tempo /= 2
    hit_thr = float(np.percentile(hits, 72))
    return prof, tempo, hit_thr


def _pts(xyz, x0, y0, w, h):
    pts = {}
    for j, i in JOINT_INDEX.items():
        x = (xyz[i, 0] + 0.8) / 1.6
        y = 1.0 - (xyz[i, 1] + 1.1) / 2.0
        pts[j] = (int(x0 + x * w), int(y0 + y * h))
    return pts


def draw_skeleton(img, cv2, xyz, x0, y0, w, h, color, thick=2):
    p = _pts(xyz, x0, y0, w, h)
    for a, b in BONES:
        cv2.line(img, p[a], p[b], color, thick, cv2.LINE_AA)
    for j in p:
        cv2.circle(img, p[j], 3, color, -1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--kind", choices=["bvh", "npz", "melody"], default="bvh")
    ap.add_argument("--out", default="out/build.mp4")
    ap.add_argument("--wav", default="out/build.wav")
    args = ap.parse_args()
    import cv2

    cfg = BuilderConfig()
    instr = cfg.order
    bars_each = cfg.count_in_bars + len(instr) * (cfg.bars_rhythm + cfg.bars_pitch) + 4
    src = energetic_window(_load(args.path, args.kind, None), 60)
    prof, tempo, hit_thr = calibrate(src)

    # how long the whole build needs, in seconds, at the locked tempo
    sec = bars_each * cfg.beats_per_bar * 60.0 / (tempo * 60)
    fps = src.fps
    src = ArrayPoseSource(src.xyz_seq[:int(sec * fps)], fps=fps, flip_y=src.flip_y,
                          name=src.name)

    be = LogBackend()
    sb = SongBuilder(be, prof, tempo, hit_thr, cfg)

    W, H = 1280, 560
    PANEL_H, TL_Y = 400, 430
    pw = W // len(instr)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    for fr in src.frames():
        ui = sb.step(fr)
        img = np.full((H, W, 3), 24, np.uint8)
        for k, role in enumerate(instr):
            x0 = k * pw
            col = COLORS[role]
            status = ui.status[role]
            cv2.rectangle(img, (x0 + 2, 30), (x0 + pw - 2, PANEL_H), (45, 45, 45), 1)
            badge = {"pending": "", "rhythm": "REC RHYTHM", "pitch": "REC PITCH",
                     "saved": "loop"}[status]
            on = role == ui.active
            cv2.putText(img, role.upper(), (x0 + 12, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col if status != "pending" else (90, 90, 90), 2)
            if badge:
                cv2.putText(img, badge, (x0 + 12, PANEL_H - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255) if on else col, 1)
            # live skeleton in the active panel; ghost in saved panels
            if on:
                draw_skeleton(img, cv2, ui.live_xyz, x0, 30, pw, PANEL_H - 40,
                              (255, 255, 255), 2)
            elif status == "saved" and ui.ghosts[role] is not None:
                draw_skeleton(img, cv2, ui.ghosts[role], x0, 30, pw, PANEL_H - 40,
                              col, 2)
        # header
        ph = ui.phase.replace("_", " ").upper()
        cv2.putText(img, f"{ph}  {ui.bars_left} bars   |   {ui.bpm:.0f} BPM   |   "
                    f"pitch height", (16, H - 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        # pitch height meter (right side)
        my = int((PANEL_H) - ui.pitch01 * (PANEL_H - 40))
        cv2.rectangle(img, (W - 18, 30), (W - 8, PANEL_H), (60, 60, 60), 1)
        cv2.rectangle(img, (W - 18, my), (W - 8, PANEL_H), (200, 200, 80), -1)
        # loop timeline: 4 rows of onset ticks + playhead
        rows = len(instr)
        for k, role in enumerate(instr):
            y = TL_Y + k * 28
            cv2.putText(img, role[:2], (4, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        COLORS[role], 1)
            n = len(ui.onsets[role])
            for s, present in enumerate(ui.onsets[role]):
                if present:
                    sx = 30 + int(s / n * (W - 60))
                    cv2.rectangle(img, (sx, y), (sx + 4, y + 18), COLORS[role], -1)
        # playhead
        px = 30 + int(ui.playhead / cfg.loop_steps * (W - 60))
        cv2.line(img, (px, TL_Y - 4), (px, TL_Y + rows * 28), (255, 255, 255), 1)
        vw.write(img)
    vw.release()
    render_wav(be.events, args.wav)
    from collections import Counter
    print(f"tempo={tempo*60:.0f}bpm  saved={[r for r in instr if sb.status[r]=='saved']}")
    print("onsets/stem:", {r: sum(ui.onsets[r]) for r in instr})
    print(f"rendered -> {args.out}  (+ {args.wav})")


if __name__ == "__main__":
    main()
