#!/usr/bin/env python3
"""Live Studio instrument: dance in front of your webcam and *be the band*.

The five-stem :class:`~everybody_dance.studio.Studio` plays continuously; your
whole body shapes every stem at once, named moves punch in fills/drops/loops, and
the gallery-grade :mod:`everybody_dance.stage` draws a glowing body, the stem
lanes, gold-move cards and an onboarding attract loop -- legible from across a
room. Audio is the zero-setup built-in synth (no DAW needed).

    python -m tools.studio_live                  # default camera, calibrate, play
    python -m tools.studio_live --no-audio       # visual only
    python -m tools.studio_live --camera 1 --mirror --coupling 0.5
    python -m tools.studio_live --mapping my_mapping.json --record out/live.mp4

Calibration: dance freely for ~N seconds while it learns your signature/tempo.

Keys (focus the video window):
    q        quit
    c        re-calibrate
    SPACE    pause / resume
    1..0     FORCE a move (handy at a desk with no room to jump):
             1 HANDS UP  2 RAISE LEFT  3 RAISE RIGHT  4 T-POSE  5 SQUAT
             6 ARMS CROSSED  7 CLAP  8 PUNCH  9 JUMP  0 STOMP

This file is interactive and won't run in CI, so the heavy imports (cv2,
mediapipe) live inside main() and it imports cleanly headless.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from everybody_dance.calib import BuildCalibrator
from everybody_dance.gestures import DEFAULT_GESTURES
from everybody_dance.pose import JOINTS, MP_INDEX, PoseFrame, _normalise, _root_y

# Number keys 1..0 -> the ten built-in moves, in the canonical order.
FORCE_KEYS = {str((i + 1) % 10): name for i, name in enumerate(DEFAULT_GESTURES)}


def open_camera(cv2, index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {index}. Try --camera 1.")
    return cap


def landmarks_to_arrays(lm, w, h):
    """MediaPipe landmarks -> (normalised xyz, visibility, pixel coords, root_y).

    ``root_y`` is the raw hip-centre vertical (torso units, up positive) that the
    hip-centring otherwise removes -- it's what makes JUMP/STOMP fire live.
    """
    xyz = np.array([[lm[MP_INDEX[j]].x, lm[MP_INDEX[j]].y, lm[MP_INDEX[j]].z]
                    for j in JOINTS], dtype=float)
    vis = np.array([lm[MP_INDEX[j]].visibility for j in JOINTS])
    px = np.column_stack([xyz[:, 0] * w, xyz[:, 1] * h])
    return _normalise(xyz), vis, px, _root_y(xyz, flip_y=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--calibrate", type=float, default=12.0,
                    help="seconds of free dancing to learn your signature")
    ap.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2],
                    help="MediaPipe model_complexity (0=fastest, 2=most accurate)")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--mirror", action="store_true", help="flip the camera image")
    ap.add_argument("--record", default=None, help="also save the window to an mp4")
    ap.add_argument("--mapping", default=None,
                    help="path to a mapping JSON (MappingConfig.from_json)")
    ap.add_argument("--coupling", type=float, default=0.4,
                    help="clock entrainment strength (StudioConfig.coupling)")
    args = ap.parse_args()

    import cv2
    import mediapipe as mp

    from everybody_dance.mapping import MappingConfig
    from everybody_dance.stage import compose_stage, render_attract
    from everybody_dance.studio import Studio, StudioConfig

    mapping = (MappingConfig.from_json(args.mapping) if args.mapping
               else MappingConfig.default())
    studio_cfg = StudioConfig(coupling=args.coupling)

    cap = open_camera(cv2, args.camera)
    pose = mp.solutions.pose.Pose(model_complexity=args.complexity,
                                  smooth_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

    backend = None
    audio_note = "audio disabled (--no-audio)"
    if not args.no_audio:
        from everybody_dance.rtaudio import RealtimeSynth
        backend = RealtimeSynth()
        audio_note = ("audio ON" if backend.ok else
                      f"audio off: {backend.error} (pip install sounddevice)")

    state = {"phase": "calib", "cal": BuildCalibrator(), "studio": None,
             "paused": False, "t0": time.time()}

    def start_studio(prof):
        if backend:
            backend.panic()
        studio = Studio(backend or _Null(), prof, mapping, studio_cfg)
        state["studio"] = studio
        state["phase"] = "play"
        # Route each stem to its (identity-kit) timbre so the live band sounds
        # like the offline render; the per-frame sync below follows morphs.
        if backend and hasattr(backend, "set_preset"):
            for stem in studio.rack:
                backend.set_preset(stem.name, stem.timbre or stem.params.timbre)

    writer = [None]                       # 1-elem list so _show can lazily fill it
    fps_ema, last = 0.0, time.time()
    win = "everybodyDance - studio live"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            now = time.time()

            t_inf = time.time()
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            infer_ms = (time.time() - t_inf) * 1000

            present = bool(res.pose_landmarks)
            xyz = vis = px = None
            root_y = 0.0
            if present:
                xyz, vis, px, root_y = landmarks_to_arrays(
                    res.pose_landmarks.landmark, w, h)
            pf = PoseFrame(t=now,
                           xyz=xyz if present else np.zeros((len(JOINTS), 3)),
                           visibility=vis if present else np.zeros(len(JOINTS)),
                           raw_present=present, root_y=root_y)

            # collect any forced-move key presses for this frame
            commands = []
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord(" "):
                state["paused"] = not state["paused"]
            if k == ord("c"):
                state.update(phase="calib", cal=BuildCalibrator(), t0=now)
            kc = chr(k) if 0 <= k < 256 else ""
            if kc in FORCE_KEYS:
                commands.append(FORCE_KEYS[kc])

            # -- drive the calibrate -> play FSM --
            ui = None
            if state["phase"] == "calib":
                if present and not state["paused"]:
                    state["cal"].observe(pf)
                left = max(0.0, args.calibrate - (now - state["t0"]))
                if left <= 0 and state["cal"].n > 10:
                    prof, _tempo, _thr = state["cal"].finalize()
                    start_studio(prof)
                else:
                    canvas = render_attract(now)
                    bar = int((1 - left / max(args.calibrate, 1e-6)) * (canvas.shape[1] - 40))
                    cv2.rectangle(canvas, (20, canvas.shape[0] - 30),
                                  (20 + bar, canvas.shape[0] - 22), (120, 220, 255), -1)
                    cv2.putText(canvas, f"calibrating... dance freely  ({left:.0f}s)",
                                (20, canvas.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (200, 200, 200), 1, cv2.LINE_AA)
                    _show(cv2, win, canvas, args, writer)
                    continue

            if state["phase"] == "play":
                if state["paused"] and state.get("_last_ui") is not None:
                    ui = state["_last_ui"]
                else:
                    out = state["studio"].step(pf, commands)
                    ui = out.ui
                    state["_last_ui"] = ui
                    if backend and hasattr(backend, "set_preset"):
                        for name, a in out.automation.items():
                            backend.set_preset(name, a.get("timbre", ""))

            dt = now - last
            last = now
            fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt if dt > 0 else 0.0)
            perf = {"fps": fps_ema, "infer_ms": infer_ms,
                    "audio": bool(backend and backend.ok), "note": audio_note,
                    "landmarks_px": px}
            canvas = compose_stage(frame, ui, perf)
            _show(cv2, win, canvas, args, writer)
    except KeyboardInterrupt:
        pass
    finally:
        if state.get("studio") is not None and backend:
            try:
                state["studio"].panic()
            except Exception:
                pass
        if backend:
            backend.panic()
            backend.close()
        if writer[0]:
            writer[0].release()
        cap.release()
        cv2.destroyAllWindows()


def _show(cv2, win, canvas, args, writer):
    """imshow + optional mp4 record; ``writer`` is a 1-elem list, filled lazily."""
    if args.record:
        if writer[0] is None:
            writer[0] = cv2.VideoWriter(
                args.record, cv2.VideoWriter_fourcc(*"mp4v"), 20,
                (canvas.shape[1], canvas.shape[0]))
        writer[0].write(canvas)
    cv2.imshow(win, canvas)


class _Null:
    """Silent backend so the visual app still runs with --no-audio."""

    def send(self, ev): pass
    def panic(self): pass
    def close(self): pass


if __name__ == "__main__":
    main()
