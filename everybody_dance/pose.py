"""Pose sources: raw joints in, normalised skeleton frames out.

A :class:`PoseFrame` is the single abstraction the rest of the pipeline sees.
Coordinates are normalised image space but with **y pointing up** (so "COM
height" and "bounce" read intuitively), origin at the hip centre, and scaled by
torso length so a tall and a short dancer land in a comparable range *before*
per-person calibration refines it further.

Two sources ship:
  * :class:`SyntheticPoseSource` -- a parametric dancer. Needs no camera, so the
    whole instrument runs headless and the coupling can be evaluated in CI.
  * :class:`MediaPipePoseSource` -- a real webcam via mediapipe (lazy import).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

import numpy as np

# The joint subset we track, in a fixed order. Names follow MediaPipe Pose.
JOINTS: List[str] = [
    "nose",
    "l_shoulder", "r_shoulder",
    "l_elbow", "r_elbow",
    "l_wrist", "r_wrist",
    "l_hip", "r_hip",
    "l_knee", "r_knee",
    "l_ankle", "r_ankle",
]
JOINT_INDEX: Dict[str, int] = {n: i for i, n in enumerate(JOINTS)}

# MediaPipe Pose landmark indices for the joints above.
MP_INDEX: Dict[str, int] = {
    "nose": 0,
    "l_shoulder": 11, "r_shoulder": 12,
    "l_elbow": 13, "r_elbow": 14,
    "l_wrist": 15, "r_wrist": 16,
    "l_hip": 23, "r_hip": 24,
    "l_knee": 25, "r_knee": 26,
    "l_ankle": 27, "r_ankle": 28,
}

# Rough segment-mass fractions used to weight kinetic energy / centre of mass.
MASS: Dict[str, float] = {
    "nose": 0.08,
    "l_shoulder": 0.05, "r_shoulder": 0.05,
    "l_elbow": 0.03, "r_elbow": 0.03,
    "l_wrist": 0.02, "r_wrist": 0.02,
    "l_hip": 0.12, "r_hip": 0.12,
    "l_knee": 0.05, "r_knee": 0.05,
    "l_ankle": 0.02, "r_ankle": 0.02,
}
MASS_VEC = np.array([MASS[j] for j in JOINTS])
MASS_VEC = MASS_VEC / MASS_VEC.sum()


@dataclass
class PoseFrame:
    t: float                 # wall-clock-ish timestamp (s)
    xyz: np.ndarray          # (J, 3), y up, hip-centred, torso-normalised
    visibility: np.ndarray   # (J,) in [0,1]
    raw_present: bool = True # False when no body was detected this frame
    # Raw global vertical of the body root (hip centre) BEFORE hip-centring,
    # in torso-length units, **up positive** (so it's body-scale invariant).
    # Hip-centring removes this from `xyz`, so it's the only place pure vertical
    # translation (jump/stomp) survives. Sources that can't measure it leave the
    # default 0.0, which simply never fires the translation gestures.
    root_y: float = 0.0

    def joint(self, name: str) -> np.ndarray:
        return self.xyz[JOINT_INDEX[name]]


def _normalise(xyz: np.ndarray, flip_y: bool = True) -> np.ndarray:
    """Hip-centre, (optionally) flip y up, scale by torso length.

    `flip_y=True` for image-space sources (MediaPipe/synthetic, y points down);
    `flip_y=False` for sources already in a y-up world frame (BVH mocap).
    """
    xyz = xyz.copy()
    if flip_y:
        xyz[:, 1] = -xyz[:, 1]  # image y is down; make it up
    hip = 0.5 * (xyz[JOINT_INDEX["l_hip"]] + xyz[JOINT_INDEX["r_hip"]])
    shoulder = 0.5 * (xyz[JOINT_INDEX["l_shoulder"]] + xyz[JOINT_INDEX["r_shoulder"]])
    xyz = xyz - hip
    torso = np.linalg.norm(shoulder - hip)
    if torso < 1e-3:
        torso = 1.0
    return xyz / torso


def _root_y(xyz: np.ndarray, flip_y: bool = True) -> float:
    """Raw hip-centre vertical BEFORE hip-centring, in torso units, up positive.

    Mirrors :func:`_normalise`'s orientation (`flip_y`) and torso scale so the
    value lives in the same units as the normalised skeleton. The absolute
    baseline is arbitrary (recognisers use its velocity), only the scale matters.
    """
    xyz = np.asarray(xyz, float)
    hip_y = 0.5 * (xyz[JOINT_INDEX["l_hip"], 1] + xyz[JOINT_INDEX["r_hip"], 1])
    if flip_y:
        hip_y = -hip_y          # image y is down; make up positive
    hip = 0.5 * (xyz[JOINT_INDEX["l_hip"]] + xyz[JOINT_INDEX["r_hip"]])
    shoulder = 0.5 * (xyz[JOINT_INDEX["l_shoulder"]] + xyz[JOINT_INDEX["r_shoulder"]])
    torso = np.linalg.norm(shoulder - hip)
    if torso < 1e-3:
        torso = 1.0
    return float(hip_y / torso)


class PoseSource:
    """Interface: iterate :class:`PoseFrame` objects in real time."""

    def frames(self) -> Iterator[PoseFrame]:
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - trivial
        pass


class SyntheticPoseSource(PoseSource):
    """A scriptable dancer for headless runs and tests.

    The dancer bobs vertically (the beat), sways, and swings arms. A `script`
    of (duration, params) segments lets tests exercise periodic motion,
    tempo changes, stillness (for the fallback / freeze path), and bursts.
    """

    def __init__(self, fps: float = 30.0, duration: float = 30.0,
                 tempo_hz: float = 2.0, energy: float = 1.0,
                 seed: int = 0, script: Optional[List[dict]] = None,
                 realtime: bool = False):
        self.fps = fps
        self.duration = duration
        self.tempo_hz = tempo_hz
        self.energy = energy
        self.realtime = realtime
        self.script = script
        self.rng = np.random.default_rng(seed)

    def _base_skeleton(self) -> np.ndarray:
        # A neutral standing pose in image-ish coords (y down), roughly normalised.
        s = {
            "nose": (0.0, -0.62, 0.0),
            "l_shoulder": (-0.18, -0.45, 0.0), "r_shoulder": (0.18, -0.45, 0.0),
            "l_elbow": (-0.26, -0.22, 0.0), "r_elbow": (0.26, -0.22, 0.0),
            "l_wrist": (-0.30, 0.0, 0.0), "r_wrist": (0.30, 0.0, 0.0),
            "l_hip": (-0.12, 0.0, 0.0), "r_hip": (0.12, 0.0, 0.0),
            "l_knee": (-0.13, 0.35, 0.0), "r_knee": (0.13, 0.35, 0.0),
            "l_ankle": (-0.14, 0.7, 0.0), "r_ankle": (0.14, 0.7, 0.0),
        }
        return np.array([s[j] for j in JOINTS], dtype=float)

    def _params_at(self, t: float) -> dict:
        p = dict(tempo_hz=self.tempo_hz, energy=self.energy,
                 openness=1.0, asym=0.0)
        if not self.script:
            return p
        acc = 0.0
        for seg in self.script:
            acc += seg.get("duration", 0.0)
            if t < acc:
                p.update({k: v for k, v in seg.items() if k != "duration"})
                return p
        # past the end of the script: hold the last segment
        last = {k: v for k, v in self.script[-1].items() if k != "duration"}
        p.update(last)
        return p

    def frames(self) -> Iterator[PoseFrame]:
        base = self._base_skeleton()
        dt = 1.0 / self.fps
        n = int(self.duration * self.fps)
        phase = 0.0
        t0 = time.time()
        for i in range(n):
            t = i * dt
            p = self._params_at(t)
            phase += 2 * np.pi * p["tempo_hz"] * dt
            e = p["energy"]
            xyz = base.copy()
            bob = 0.10 * e * np.sin(phase)              # vertical beat
            sway = 0.06 * e * np.sin(0.5 * phase)        # slower lateral sway
            arm = 0.22 * e * p["openness"]
            # Knee-flex bounce: the upper body drops toward planted feet, so the
            # COM-above-feet height oscillates (a rigid whole-body bob would be
            # cancelled by hip-centring downstream). Feet stay put, knees half.
            base_ankle_y = base[[JOINT_INDEX["l_ankle"], JOINT_INDEX["r_ankle"]], 1].copy()
            base_knee_y = base[[JOINT_INDEX["l_knee"], JOINT_INDEX["r_knee"]], 1].copy()
            xyz[:, 1] -= bob
            xyz[[JOINT_INDEX["l_ankle"], JOINT_INDEX["r_ankle"]], 1] = base_ankle_y
            xyz[[JOINT_INDEX["l_knee"], JOINT_INDEX["r_knee"]], 1] = base_knee_y - 0.5 * bob
            xyz[:, 0] += sway
            # Arms swing in counter-phase, modulated by openness.
            for side, sgn in (("l", 1.0), ("r", -1.0)):
                wi = JOINT_INDEX[f"{side}_wrist"]
                ei = JOINT_INDEX[f"{side}_elbow"]
                swing = arm * np.sin(phase + (0 if side == "l" else np.pi))
                asy = 1.0 + (p["asym"] if side == "l" else -p["asym"])
                xyz[wi, 1] -= swing * asy
                xyz[wi, 0] += 0.10 * sgn * arm * np.cos(phase) * asy
                xyz[ei, 1] -= 0.5 * swing * asy
            xyz += self.rng.normal(0, 0.002, xyz.shape)  # sensor jitter
            vis = np.ones(len(JOINTS))
            yield PoseFrame(t=t, xyz=_normalise(xyz), visibility=vis)
            if self.realtime:
                target = t0 + (i + 1) * dt
                slack = target - time.time()
                if slack > 0:
                    time.sleep(slack)


class MediaPipePoseSource(PoseSource):
    """Real webcam via MediaPipe Pose (lazy import; needs `mediapipe`,`opencv`)."""

    def __init__(self, camera: int = 0, model_complexity: int = 1):
        self.camera = camera
        self.model_complexity = model_complexity
        self._cap = None
        self._pose = None

    def _ensure(self):
        if self._pose is not None:
            return
        import cv2  # noqa
        import mediapipe as mp  # noqa
        self._cv2 = cv2
        self._cap = cv2.VideoCapture(self.camera)
        self._pose = mp.solutions.pose.Pose(
            model_complexity=self.model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def frames(self) -> Iterator[PoseFrame]:
        self._ensure()
        cv2 = self._cv2
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            t = time.time()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self._pose.process(rgb)
            if not res.pose_landmarks:
                yield PoseFrame(t=t, xyz=np.zeros((len(JOINTS), 3)),
                                visibility=np.zeros(len(JOINTS)),
                                raw_present=False)
                continue
            lm = res.pose_landmarks.landmark
            xyz = np.array([[lm[MP_INDEX[j]].x, lm[MP_INDEX[j]].y, lm[MP_INDEX[j]].z]
                            for j in JOINTS], dtype=float)
            vis = np.array([lm[MP_INDEX[j]].visibility for j in JOINTS])
            yield PoseFrame(t=t, xyz=_normalise(xyz), visibility=vis,
                            root_y=_root_y(xyz, flip_y=True))

    def close(self) -> None:  # pragma: no cover
        if self._cap is not None:
            self._cap.release()
