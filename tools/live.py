#!/usr/bin/env python3
"""Live sandbox: dance in front of your webcam, watch the pose tracking, the
song-builder UI, and real-time performance -- and hear the song build (built-in
synth, no DAW needed).

    python -m tools.live                 # default camera, ~15s calibration
    python -m tools.live --no-audio      # visual only
    python -m tools.live --camera 1 --calibrate 10 --mirror

Keys (focus the video window):  q quit  ·  r restart song (keep calibration)
                                c re-calibrate  ·  SPACE pause/resume
"""

import argparse
import time

import numpy as np

from everybody_dance.builder import SongBuilder, BuilderConfig
from everybody_dance.calib import BuildCalibrator
from everybody_dance.pose import JOINTS, MP_INDEX, PoseFrame, _normalise
from everybody_dance.viz import COLORS, FONT, compose_live


def open_camera(cv2, index, width=960):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {index}. Try --camera 1.")
    return cap


def landmarks_to_arrays(lm, w, h):
    """MediaPipe landmarks -> (normalised xyz for the pipeline, pixel coords)."""
    xyz = np.array([[lm[MP_INDEX[j]].x, lm[MP_INDEX[j]].y, lm[MP_INDEX[j]].z]
                    for j in JOINTS], dtype=float)
    vis = np.array([lm[MP_INDEX[j]].visibility for j in JOINTS])
    px = np.column_stack([xyz[:, 0] * w, xyz[:, 1] * h])
    return _normalise(xyz), vis, px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--calibrate", type=float, default=15.0)
    ap.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2],
                    help="MediaPipe model_complexity (0=fastest, 2=most accurate)")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--mirror", action="store_true", help="flip the camera image")
    ap.add_argument("--record", default=None, help="also save the window to an mp4")
    args = ap.parse_args()

    import cv2
    import mediapipe as mp

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
    cfg = BuilderConfig()

    # gesture layer: recognised moves -> one-shot effects + on-screen flash,
    # layered over the continuous song-builder.
    from everybody_dance.effects import EffectEngine
    from everybody_dance.features import FeatureExtractor
    from everybody_dance.gestures import GestureRecognizer, default_gestures
    gfe = FeatureExtractor()
    rec = GestureRecognizer(default_gestures())

    state = {"phase": "calib", "cal": BuildCalibrator(), "sb": None, "fx": None,
             "prof": None, "tempo": None, "thr": None, "paused": False,
             "t0": time.time()}

    def start_song():
        if backend:
            backend.panic()
        state["sb"] = SongBuilder(backend or _Null(), state["prof"],
                                  state["tempo"], state["thr"], cfg)
        state["fx"] = EffectEngine(backend or _Null(), state["sb"].sub)
        state["phase"] = "build"

    writer = None
    fps_ema, last = 0.0, time.time()
    win = "everybodyDance - live"
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

            xyz, vis, px = (None, None, None)
            present = bool(res.pose_landmarks)
            if present:
                xyz, vis, px = landmarks_to_arrays(res.pose_landmarks.landmark, w, h)
            pf = PoseFrame(t=now, xyz=xyz if present else np.zeros((len(JOINTS), 3)),
                           visibility=vis if present else np.zeros(len(JOINTS)),
                           raw_present=present)

            # --- drive the state machine ---
            if state["phase"] == "calib":
                if present and not state["paused"]:
                    state["cal"].observe(pf)
                left = max(0.0, args.calibrate - (now - state["t0"]))
                if left <= 0 and state["cal"].n > 10:
                    state["prof"], state["tempo"], state["thr"] = state["cal"].finalize()
                    start_song()
                ui = _calib_ui(cfg, left, xyz)
            elif state["paused"] and state.get("_last_ui") is not None:
                ui = state["_last_ui"]              # frozen frame while paused
            else:
                ui = state["sb"].step(pf)
                state["_last_ui"] = ui

            # --- gesture layer (active once the song is building) ---
            flashes = []
            if state["fx"] is not None and present and not state["paused"]:
                gfeats = gfe.update(pf)
                for ev in rec.update(pf, gfeats):
                    state["fx"].trigger(ev.name, now)
                flashes = state["fx"].active_flashes(now)

            # --- compose & show ---
            t_draw = time.time()
            dt = now - last
            last = now
            fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt if dt > 0 else 0)
            perf = {"fps": fps_ema, "infer_ms": infer_ms, "audio": bool(backend and backend.ok),
                    "landmarks_px": px, "note": audio_note, "draw_ms": 0.0,
                    "flashes": flashes, "t": now}
            canvas = compose_live(frame, ui, perf, cfg.loop_steps)
            perf_draw = (time.time() - t_draw) * 1000
            cv2.putText(canvas, f"draw {perf_draw:.0f}ms", (300, 78), FONT, 0.5,
                        (170, 170, 170), 1)
            if not present:
                cv2.putText(canvas, "no person detected", (20, 470), FONT, 0.7,
                            (60, 60, 230), 2)

            if args.record:
                if writer is None:
                    writer = cv2.VideoWriter(args.record,
                                             cv2.VideoWriter_fourcc(*"mp4v"), 20,
                                             (canvas.shape[1], canvas.shape[0]))
                writer.write(canvas)
            cv2.imshow(win, canvas)

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord(" "):
                state["paused"] = not state["paused"]
            if k == ord("r") and state["prof"] is not None:
                start_song()
            if k == ord("c"):
                state.update(phase="calib", cal=BuildCalibrator(), t0=time.time())
    except KeyboardInterrupt:
        pass
    finally:
        if backend:
            backend.panic()
            backend.close()
        if writer:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()


def _calib_ui(cfg, left, xyz):
    from everybody_dance.builder import UIState
    z = np.zeros((len(JOINTS), 3)) if xyz is None else xyz
    return UIState(order=list(cfg.order),
                   status={r: "pending" for r in cfg.order}, active=None,
                   phase="calibrating", bars_left=int(left),
                   playhead=0, onsets={r: [False] * cfg.loop_steps for r in cfg.order},
                   pitch01=0.5, live_xyz=z, ghosts={r: None for r in cfg.order}, bpm=0.0)


class _Null:
    def send(self, ev): pass
    def panic(self): pass
    def close(self): pass


if __name__ == "__main__":
    main()
