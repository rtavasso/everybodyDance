#!/usr/bin/env python3
"""Run MediaPipe Pose on a video and save per-frame keypoints.

Output: an .npz with
    landmarks : (T, 33, 4)  -> x, y, z (image-normalised), visibility
    present   : (T,)        -> bool, was a body detected this frame
    fps       : float

This is the "actual pose estimation data" the rest of the iteration runs on.
"""
import argparse
import os
import sys

import numpy as np


def extract(path, max_seconds=None, stride=1, model_complexity=1,
            model_path="models/pose_landmarker.task"):
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    landmarker = vision.PoseLandmarker.create_from_options(options)

    lms, present = [], []
    i = 0
    max_frames = int(max_seconds * fps) if max_seconds else None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames and i >= max_frames:
            break
        if i % stride != 0:
            i += 1
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(i / fps * 1000)
        res = landmarker.detect_for_video(mp_image, ts_ms)
        if res.pose_landmarks:
            lm = res.pose_landmarks[0]   # most prominent person
            arr = np.array([[p.x, p.y, p.z, p.visibility] for p in lm], dtype=np.float32)
            lms.append(arr)
            present.append(True)
        else:
            lms.append(np.zeros((33, 4), dtype=np.float32))
            present.append(False)
        i += 1
    cap.release()
    landmarker.close()
    return np.array(lms), np.array(present), fps / stride


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--complexity", type=int, default=1)
    args = ap.parse_args()

    lms, present, fps = extract(args.video, args.max_seconds, args.stride, args.complexity)
    np.savez_compressed(args.out, landmarks=lms, present=present, fps=fps)

    # MediaPipe Pose landmark indices for the joints the engine tracks.
    key = {"nose": 0, "l_sh": 11, "r_sh": 12, "l_hip": 23, "r_hip": 24,
           "l_wr": 15, "r_wr": 16, "l_ank": 27, "r_ank": 28}
    det = present.mean() if len(present) else 0.0
    print(f"{os.path.basename(args.video)}: {len(lms)} frames @ {fps:.1f}fps, "
          f"detected {det*100:.0f}%")
    if det > 0:
        vis = lms[present][:, :, 3]  # (Tdet, 33)
        line = "  visibility: " + " ".join(
            f"{k}={vis[:, idx].mean():.2f}" for k, idx in key.items())
        print(line)
    print(f"  saved -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
