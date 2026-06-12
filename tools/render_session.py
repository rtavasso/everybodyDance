#!/usr/bin/env python3
"""Run a REAL dance video through the whole instrument and film the result.

The full gallery experience, offline: pose-estimated keypoints (from
tools/extract_pose.py) drive the Studio with the game layer live -- the
recognizer fires whatever moves the real dancer actually makes (no scripted
commands), the song arc unlocks stems as they earn heat, Dance DNA picks their
sonic world -- and every frame is rendered through the gallery Stage with the
original video as the dim backdrop. Output:

    session.mp4   the stage film (muxed with audio when ffmpeg is available)
    music.wav     the stereo, sidechained, bus-compressed render
    pianoroll.png + metrics.json + a stills strip

    uv run python -m tools.render_session data/poses/benchmark_dance.npz \
        --video data/videos/benchmark_dance.mp4 --out out/session

Deterministic end to end: the same npz replays to byte-identical events/audio.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from typing import List, Optional, Tuple

import numpy as np

from everybody_dance import metrics as M
from everybody_dance.features import FeatureExtractor
from everybody_dance.gestures import GESTURE_LIBRARY
from everybody_dance.output import LogBackend
from everybody_dance.pose import JOINTS
from everybody_dance.sources import MP_SUBSET, load_bvh_source, load_npz_source
from everybody_dance.studio import Studio, StudioConfig
from tools.render_gestures import make_profile
from tools.render_studio import (compute_metrics, multistem_pianoroll,
                                 render_wav)

W, H = 1280, 720


def load_image_landmarks(pose_path: str):
    """The raw image-space (x, y, visibility) trace for the stage overlay --
    this is what lets the film draw the glowing skeleton ON the dancer."""
    if not pose_path.endswith(".npz"):
        return None
    d = np.load(pose_path)
    sel = [MP_SUBSET[j] for j in JOINTS]
    lm = d["landmarks"][:, sel]
    return lm[:, :, :2], lm[:, :, 3]            # (T,13,2) xy01, (T,13) vis


def draw_pose_overlay(cam, xy01, vis, box, color):
    """Glow-trace the estimated skeleton in the dancer's true image position
    (inside the letterbox `box` = (x0, y0, w, h)). Joints under-confidence are
    skipped, so partial detections degrade gracefully."""
    import cv2

    from everybody_dance import viz
    from everybody_dance.pose import JOINT_INDEX
    x0, y0, bw, bh = box
    pts = np.column_stack([x0 + xy01[:, 0] * bw, y0 + xy01[:, 1] * bh])
    pts = pts.astype(int)
    bright = tuple(int(np.clip(c * 1.5, 0, 255)) for c in color)
    ov = cam.copy()
    for a, b in viz.BONES:
        ia, ib = JOINT_INDEX[a], JOINT_INDEX[b]
        if vis[ia] < 0.4 or vis[ib] < 0.4:
            continue
        cv2.line(ov, tuple(pts[ia]), tuple(pts[ib]), bright, 12, cv2.LINE_AA)
    cv2.addWeighted(ov, 0.35, cam, 0.65, 0, cam)
    for a, b in viz.BONES:
        ia, ib = JOINT_INDEX[a], JOINT_INDEX[b]
        if vis[ia] < 0.4 or vis[ib] < 0.4:
            continue
        cv2.line(cam, tuple(pts[ia]), tuple(pts[ib]), bright, 3, cv2.LINE_AA)
        cv2.circle(cam, tuple(pts[ib]), 4, bright, -1, cv2.LINE_AA)


def load_source(path: str, kind: Optional[str], max_seconds: Optional[float]):
    kind = kind or ("bvh" if path.endswith(".bvh") else "npz")
    if kind == "bvh":
        return load_bvh_source(path, max_seconds=max_seconds)
    return load_npz_source(path)


def letterbox(frame, w: int, h: int):
    """Fit a camera frame into (w, h) preserving aspect (pad with black), so a
    portrait phone video isn't smeared across the landscape stage. Returns the
    padded image and the (x0, y0, nw, nh) box the frame landed in."""
    import cv2
    fh, fw = frame.shape[:2]
    scale = min(w / fw, h / fh)
    nw, nh = max(int(fw * scale), 2), max(int(fh * scale), 2)
    resized = cv2.resize(frame, (nw, nh))
    out = np.zeros((h, w, 3), np.uint8)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = resized
    return out, (x0, y0, nw, nh)


def run_session(pose_path: str, video: Optional[str], out_dir: str,
                kind: Optional[str] = None, max_seconds: Optional[float] = None,
                coupling: float = 0.4, write_video: bool = True) -> dict:
    import cv2

    from everybody_dance.stage import Stage

    os.makedirs(out_dir, exist_ok=True)
    src = load_source(pose_path, kind, max_seconds)
    frames = list(src.frames())
    fps = src.fps
    dur = len(frames) / fps
    name = getattr(src, "name", os.path.basename(pose_path))

    # Calibrate on the whole clip (it's a dance throughout), like run_dataset.
    profile = make_profile(iter(frames), fps)

    be = LogBackend()
    studio = Studio(be, profile, cfg=StudioConfig(coupling=coupling))
    stage = Stage(W, H)
    fe = FeatureExtractor(fps)

    cap = cv2.VideoCapture(video) if video else None
    writer = None
    video_path = os.path.join(out_dir, "stage_video.mp4")
    if write_video:
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (W, H))

    stage_w = int(W * 0.62)
    autom: List[Tuple[float, dict]] = []
    fired: List[Tuple[float, str]] = []
    proc_ms: List[float] = []
    mv = {k: [] for k in ("energy", "core", "limb", "open", "comh", "norm_e", "gated")}
    segments = [(0.0, studio.sub.cfg.tonic, studio.sub.cfg.scale)]
    stills = []
    per_frame = []

    img_lm = load_image_landmarks(pose_path)
    for i, fr in enumerate(frames):
        cam = box = None
        if cap is not None:
            ok, raw = cap.read()
            if ok:
                cam, box = letterbox(raw, stage_w, H)
        t0 = time.perf_counter()
        out = studio.step(fr, [])
        proc_ms.append((time.perf_counter() - t0) * 1000.0)
        if cam is not None and img_lm is not None and i < len(img_lm[0]) \
                and getattr(fr, "raw_present", True):
            from everybody_dance.stage import mood_color
            draw_pose_overlay(cam, img_lm[0][i], img_lm[1][i], box,
                              mood_color(out.ui.mood))

        f = fe.update(fr)
        mv["energy"].append(f.energy_env)
        mv["core"].append(f.core_energy_env)
        mv["limb"].append(f.limb_energy_env)
        mv["open"].append(f.openness)
        mv["comh"].append(f.com_height)
        mv["norm_e"].append(float(out.ui.energy))
        mv["gated"].append(1.0 if out.ui.still else 0.0)
        if (studio.sub.cfg.tonic, studio.sub.cfg.scale) != segments[-1][1:]:
            segments.append((fr.t, studio.sub.cfg.tonic, studio.sub.cfg.scale))
        autom.append((fr.t, out.automation))
        per_frame.append((out.ui.live_xyz, list(out.ui.flashes)))
        seen = set()
        for fl in out.ui.flashes:
            if (abs(fl.t0 - fr.t) < 1e-9 and fl.name in GESTURE_LIBRARY
                    and fl.name not in seen):
                seen.add(fl.name)
                fired.append((fr.t, fl.name))

        if writer is not None:
            has_overlay = cam is not None and img_lm is not None
            canvas = stage.compose(cam, out.ui,
                                   {"fps": fps, "infer_ms": 0.0, "audio": True,
                                    "note": name},
                                   draw_body=not has_overlay,
                                   cam_gain=0.85 if has_overlay else 0.32)
            writer.write(canvas)
            if i in {int(k * (len(frames) - 1) / 5) for k in range(6)}:
                stills.append(canvas.copy())
    studio.panic()
    if writer is not None:
        writer.release()
    if cap is not None:
        cap.release()

    # -- audio + artifacts ---------------------------------------------------
    wav_path = os.path.join(out_dir, "music.wav")
    render_wav(be.events, autom, wav_path, dur)

    r = dict(studio=studio, events=list(be.events), fired=fired,
             per_frame=per_frame, proc_ms=proc_ms, mv=mv, fps=fps, dur=dur,
             autom=autom, sub=studio.sub, segments=segments)
    metrics = compute_metrics(r, labels=None)
    slo = M.check_slo({k: v for k, v in _flat(metrics).items()})
    json.dump(metrics, open(os.path.join(out_dir, "metrics.json"), "w"),
              indent=2)
    cv2.imwrite(os.path.join(out_dir, "pianoroll.png"),
                multistem_pianoroll(be.events, dur))
    if stills:
        cv2.imwrite(os.path.join(out_dir, "stills.png"), np.vstack(stills[:4]))

    final = mux(video_path, wav_path, os.path.join(out_dir, "session.mp4")) \
        if write_video else None

    return dict(metrics=metrics, slo=slo, events=be.events, studio=studio,
                fired=fired, out=out_dir, final=final, dur=dur)


def _flat(metrics) -> dict:
    from tools.render_studio import flatten
    return flatten(metrics)


def mux(video_path: str, wav_path: str, out_path: str) -> Optional[str]:
    """Mux the stage video with the rendered audio via imageio-ffmpeg's bundled
    binary (optional dependency); returns the muxed path or None."""
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        print("(!) imageio-ffmpeg not installed -- leaving video + wav unmuxed "
              "(uv pip install imageio-ffmpeg)")
        return None
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", video_path,
           "-i", wav_path, "-map", "0:v", "-map", "1:a",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22",
           "-c:a", "aac", "-b:a", "160k", "-shortest", out_path]
    try:
        subprocess.run(cmd, check=True)
        return out_path
    except Exception as e:
        print(f"(!) mux failed ({e}); video + wav left separate")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pose", help=".npz (extract_pose) or .bvh pose file")
    ap.add_argument("--video", default=None,
                    help="original video for the stage backdrop (frame-synced)")
    ap.add_argument("--kind", choices=["npz", "bvh"], default=None)
    ap.add_argument("--out", default="out/session")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--coupling", type=float, default=0.4)
    args = ap.parse_args()

    res = run_session(args.pose, args.video, args.out, kind=args.kind,
                      max_seconds=args.max_seconds, coupling=args.coupling)
    m, st = res["metrics"], res["studio"]
    g = m["game"]
    print(f"\n== {os.path.basename(args.pose)} ({res['dur']:.1f}s) ==")
    if "identity" in g:
        i = g["identity"]
        print(f"DANCE DNA: {i['name']} -- {i['root']} {i['scale']} - "
              f"{i['kit']} kit, progression {i['progression']}")
    print(f"arc: {' -> '.join(g['sections'])} | stems unlocked: "
          f"{g['stems_unlocked']}/5 | streak tier max: {g['max_streak_tier']}")
    print(f"combos: {g['combos'] or 'none'} | gold: {g['gold_hits']}/"
          f"{g['gold_prompts']} hit")
    rec = m["recognition"]
    print(f"moves fired by the real dancer: {rec.get('fires', {})}")
    print("COUPLING:", json.dumps(m["coupling"]["score"]),
          "| phantom:", m["coupling"]["phantom"],
          "| in-scale:", m["musicality"]["in_scale_pct"], "%")
    print("SLO:", json.dumps(res["slo"]))
    print(f"\nbundle -> {res['out']}/"
          + (f" (session.mp4 muxed)" if res["final"] else
             " (stage_video.mp4 + music.wav, unmuxed)"))


if __name__ == "__main__":
    main()
