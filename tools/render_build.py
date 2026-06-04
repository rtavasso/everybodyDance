#!/usr/bin/env python3
"""Render the song-builder offline: dance continuously, the song builds, and each
committed stem becomes a looping skeleton ghost. Outputs an overlay MP4 (the
4-panel chorus + loop timeline) and the WAV it produces.
"""

import argparse
import os

import numpy as np

from everybody_dance.builder import SongBuilder, BuilderConfig
from everybody_dance.calib import calibrate_build
from everybody_dance.sources import ArrayPoseSource
from everybody_dance.viz import compose_offline
from tools.run_dataset import _load
from tools.synth import render as render_wav


def energetic_window(src, seconds):
    from everybody_dance.features import FeatureExtractor
    fe = FeatureExtractor(src.fps)
    vals = np.array([(f.core_energy_env + f.limb_energy_env) for f in
                     (fe.update(fr) for fr in src.frames())])
    W = int(seconds * src.fps)
    if len(vals) <= W:
        return src
    start = int(np.argmax(np.convolve(vals, np.ones(W) / W, "valid")))
    return ArrayPoseSource(src.xyz_seq[start:start + W], fps=src.fps,
                           flip_y=src.flip_y, name=src.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--kind", choices=["bvh", "npz", "melody"], default="bvh")
    ap.add_argument("--out", default="out/build.mp4")
    ap.add_argument("--wav", default="out/build.wav")
    args = ap.parse_args()
    import cv2

    cfg = BuilderConfig()
    bars = cfg.count_in_bars + len(cfg.order) * (cfg.bars_rhythm + cfg.bars_pitch) + 4
    src = energetic_window(_load(args.path, args.kind, None), 60)
    prof, tempo, hit_thr = calibrate_build(src)

    sec = bars * cfg.beats_per_bar * 60.0 / (tempo * 60)
    fps = src.fps
    src = ArrayPoseSource(src.xyz_seq[:int(sec * fps)], fps=fps, flip_y=src.flip_y,
                          name=src.name)

    from everybody_dance.output import LogBackend
    be = LogBackend()
    sb = SongBuilder(be, prof, tempo, hit_thr, cfg)

    W, H = 1280, 560
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for fr in src.frames():
        ui = sb.step(fr)
        vw.write(compose_offline(ui, W, H, cfg.loop_steps))
    vw.release()
    render_wav(be.events, args.wav)
    print(f"tempo={tempo*60:.0f}bpm  saved={[r for r in cfg.order if sb.status[r]=='saved']}")
    print("onsets/stem:", {r: sum(s is not None for s in sb.loops[r]) for r in cfg.order})
    print(f"rendered -> {args.out}  (+ {args.wav})")


if __name__ == "__main__":
    main()
