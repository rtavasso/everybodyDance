"""File-based pose sources: real motion-capture (BVH) and pre-extracted
MediaPipe keypoints (.npz). Both normalise into the same :class:`PoseFrame`
the engine consumes, so the instrument is agnostic to where the body came from.
"""

from __future__ import annotations

import time
from typing import Iterator, Optional

import numpy as np

from .pose import JOINTS, PoseFrame, PoseSource, _normalise

# --- joint maps -----------------------------------------------------------

# LAFAN1 (Ubisoft) BVH skeleton -> our 13-joint subset.
BVH_MAP = {
    "nose": "Head",
    "l_shoulder": "LeftArm", "r_shoulder": "RightArm",
    "l_elbow": "LeftForeArm", "r_elbow": "RightForeArm",
    "l_wrist": "LeftHand", "r_wrist": "RightHand",
    "l_hip": "LeftUpLeg", "r_hip": "RightUpLeg",
    "l_knee": "LeftLeg", "r_knee": "RightLeg",
    "l_ankle": "LeftFoot", "r_ankle": "RightFoot",
}

# MediaPipe Pose 33-landmark indices -> our subset (mirrors pose.MP_INDEX).
MP_SUBSET = {
    "nose": 0, "l_shoulder": 11, "r_shoulder": 12, "l_elbow": 13, "r_elbow": 14,
    "l_wrist": 15, "r_wrist": 16, "l_hip": 23, "r_hip": 24, "l_knee": 25,
    "r_knee": 26, "l_ankle": 27, "r_ankle": 28,
}


class ArrayPoseSource(PoseSource):
    """Wrap a pre-loaded (T, 13, 3) sequence (already in our joint order).

    `flip_y` matches :func:`_normalise` (False for y-up world frames like BVH).
    `visibility` is optional (T, 13); defaults to all-visible.
    """

    def __init__(self, xyz_seq: np.ndarray, fps: float,
                 visibility: Optional[np.ndarray] = None,
                 present: Optional[np.ndarray] = None,
                 flip_y: bool = False, realtime: bool = False,
                 name: str = "array"):
        self.xyz_seq = np.asarray(xyz_seq, dtype=float)
        self.fps = float(fps)
        self.visibility = visibility
        self.present = present
        self.flip_y = flip_y
        self.realtime = realtime
        self.name = name

    def frames(self) -> Iterator[PoseFrame]:
        dt = 1.0 / self.fps
        t0 = time.time()
        T = len(self.xyz_seq)
        for i in range(T):
            t = i * dt
            present = True if self.present is None else bool(self.present[i])
            vis = (np.ones(len(JOINTS)) if self.visibility is None
                   else self.visibility[i])
            if not present:
                yield PoseFrame(t=t, xyz=np.zeros((len(JOINTS), 3)),
                                visibility=np.zeros(len(JOINTS)), raw_present=False)
            else:
                xyz = _normalise(self.xyz_seq[i], flip_y=self.flip_y)
                yield PoseFrame(t=t, xyz=xyz, visibility=vis)
            if self.realtime:
                slack = t0 + (i + 1) * dt - time.time()
                if slack > 0:
                    time.sleep(slack)


def load_bvh_source(path: str, max_seconds: Optional[float] = None,
                    target_fps: Optional[float] = None,
                    realtime: bool = False) -> ArrayPoseSource:
    """Parse a BVH file, run FK, map to our subset, return an ArrayPoseSource."""
    from tools.bvh import BVH  # local tool, only needed for offline runs

    bvh = BVH.parse(path)
    max_frames = int(max_seconds * bvh.fps) if max_seconds else None
    world = bvh.forward_kinematics(max_frames=max_frames)   # (T, J, 3), y-up
    name_idx = {n: k for k, n in enumerate(bvh.names)}
    sel = [name_idx[BVH_MAP[j]] for j in JOINTS]
    xyz = world[:, sel, :]                                  # (T, 13, 3)

    fps = bvh.fps
    if target_fps and abs(target_fps - fps) > 1e-3:
        # simple frame resample to a target fps
        T = xyz.shape[0]
        new_T = int(T * target_fps / fps)
        idx = np.clip((np.arange(new_T) * fps / target_fps).astype(int), 0, T - 1)
        xyz = xyz[idx]
        fps = target_fps
    import os
    return ArrayPoseSource(xyz, fps=fps, flip_y=False, realtime=realtime,
                           name=os.path.splitext(os.path.basename(path))[0])


def load_npz_source(path: str, min_visibility: float = 0.3,
                    realtime: bool = False) -> ArrayPoseSource:
    """Load MediaPipe keypoints (.npz from tools/extract_pose.py)."""
    import os
    d = np.load(path)
    lm = d["landmarks"]            # (T, 33, 4): x,y,z,visibility (image coords)
    present = d["present"]
    fps = float(d["fps"])
    sel = [MP_SUBSET[j] for j in JOINTS]
    xyz = lm[:, sel, :3].astype(float)
    vis = lm[:, sel, 3].astype(float)
    # Frames with too few confident joints are treated as "no body".
    enough = (vis > min_visibility).sum(1) >= 6
    present = present & enough
    return ArrayPoseSource(xyz, fps=fps, visibility=vis, present=present,
                           flip_y=True, realtime=realtime,
                           name=os.path.splitext(os.path.basename(path))[0])
