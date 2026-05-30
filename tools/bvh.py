"""Minimal BVH parser + forward kinematics → world-space joint positions.

Used to turn real motion-capture (LAFAN1: multiple subjects, dance & locomotion,
no occlusion, full body) into clean pose sequences for the engine. Returns world
positions per frame plus the joint-name list, so a source adapter can map the
BVH skeleton onto the engine's 13-joint subset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Joint:
    name: str
    offset: np.ndarray
    channels: List[str]
    parent: Optional[int]
    children: List[int] = field(default_factory=list)
    is_end: bool = False


def _rot(axis: str, deg: float) -> np.ndarray:
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    if axis == "X":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "Y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class BVH:
    def __init__(self, joints: List[Joint], frames: np.ndarray, frame_time: float):
        self.joints = joints
        self.frames = frames                # (T, total_channels)
        self.frame_time = frame_time
        self.fps = 1.0 / frame_time if frame_time > 0 else 30.0
        self.names = [j.name for j in joints if not j.is_end]

    @staticmethod
    def parse(path: str) -> "BVH":
        with open(path) as fh:
            tokens = fh.read().split()
        it = iter(tokens)

        joints: List[Joint] = []
        stack: List[int] = []

        def read_joint(name: str, is_end: bool):
            idx = len(joints)
            parent = stack[-1] if stack else None
            joints.append(Joint(name, np.zeros(3), [], parent, [], is_end))
            if parent is not None:
                joints[parent].children.append(idx)
            return idx

        tok = next(it)
        assert tok == "HIERARCHY"
        while True:
            tok = next(it)
            if tok in ("ROOT", "JOINT"):
                name = next(it)
                idx = read_joint(name, False)
                assert next(it) == "{"
                stack.append(idx)
            elif tok == "End":
                next(it)  # "Site"
                idx = read_joint("End", True)
                assert next(it) == "{"
                stack.append(idx)
            elif tok == "OFFSET":
                ox, oy, oz = float(next(it)), float(next(it)), float(next(it))
                joints[stack[-1]].offset = np.array([ox, oy, oz])
            elif tok == "CHANNELS":
                n = int(next(it))
                joints[stack[-1]].channels = [next(it) for _ in range(n)]
            elif tok == "}":
                stack.pop()
            elif tok == "MOTION":
                break

        assert next(it) == "Frames:"
        nframes = int(next(it))
        assert next(it) == "Frame" and next(it) == "Time:"
        frame_time = float(next(it))
        total = sum(len(j.channels) for j in joints)
        data = np.array([float(next(it)) for _ in range(nframes * total)])
        frames = data.reshape(nframes, total)
        return BVH(joints, frames, frame_time)

    def forward_kinematics(self, max_frames: Optional[int] = None) -> np.ndarray:
        """Return world positions (T, J, 3) for non-end joints, in BVH units."""
        frames = self.frames if max_frames is None else self.frames[:max_frames]
        T = len(frames)
        non_end = [i for i, j in enumerate(self.joints) if not j.is_end]
        name_pos = {idx: k for k, idx in enumerate(non_end)}
        out = np.zeros((T, len(non_end), 3))

        # channel offsets per joint
        ch_off = {}
        c = 0
        for i, j in enumerate(self.joints):
            ch_off[i] = c
            c += len(j.channels)

        for t in range(T):
            world_pos: Dict[int, np.ndarray] = {}
            world_rot: Dict[int, np.ndarray] = {}
            for i, j in enumerate(self.joints):
                off = ch_off[i]
                pos = j.offset.astype(float).copy()
                R = np.eye(3)
                trans = np.zeros(3)
                ci = 0
                for ch in j.channels:
                    val = frames[t, off + ci]
                    ci += 1
                    if ch.endswith("position"):
                        ax = {"X": 0, "Y": 1, "Z": 2}[ch[0]]
                        trans[ax] = val
                    else:  # rotation
                        R = R @ _rot(ch[0], val)
                if j.parent is None:
                    world_pos[i] = pos + trans
                    world_rot[i] = R
                else:
                    pp = world_pos[j.parent]
                    pr = world_rot[j.parent]
                    world_pos[i] = pp + pr @ pos
                    world_rot[i] = pr @ R
                if not j.is_end:
                    out[t, name_pos[i]] = world_pos[i]
        return out
