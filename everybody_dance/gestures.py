"""Gesture recognition -- named, discrete dance moves that fire their own effects,
layered ON TOP of the open-ended continuous mapping.

This is how a Just-Dance-style "this specific move did this specific thing"
coexists with "all your movement keeps shaping the music". Two recognizer kinds,
both deterministic (no training, no RNG -> learnable + repeatable):

  * StaticPose / HeuristicMotion -- geometric predicates over the live skeleton
    and its velocity (hands up, T-pose, squat, clap, jump, punch, stomp). Robust,
    zero-shot, instant.
  * DTWGesture -- Dynamic Time Warping against a *recorded reference* trajectory.
    This is the closest analog to how Just Dance scores you: it compares your
    motion to a known move template. You can record your own ("teach a move").

Recognised moves are emitted as GestureEvents; an effects layer turns them into
one-shot musical hits + an on-screen flash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .pose import JOINT_INDEX


@dataclass
class GestureEvent:
    name: str
    t: float
    score: float = 1.0       # 1 = exact (pose), or 1-dist/threshold (DTW)
    kind: str = "pose"


class PoseCtx:
    """Per-frame view handed to predicates: normalised skeleton + velocity.

    The skeleton is hip-centred and torso-normalised (torso length = 1), so
    predicates use torso units and joint *relationships* (which survive the
    normalisation) rather than absolute screen position. `vnorm` is verticality
    divided by its running standing maximum, so "crouch" is person-independent.
    """

    def __init__(self, xyz, vel, feats, t, vnorm=1.0):
        self.xyz, self._vel, self.feats, self.t = xyz, vel, feats, t
        self.vnorm = vnorm

    def p(self, name) -> np.ndarray:
        return self.xyz[JOINT_INDEX[name]]

    def v(self, name) -> np.ndarray:
        return self._vel[JOINT_INDEX[name]]


# -- gesture types ---------------------------------------------------------

class Gesture:
    name: str
    cooldown: float = 0.6

    def update(self, c: PoseCtx) -> Optional[GestureEvent]:
        raise NotImplementedError


class StaticPose(Gesture):
    """Fires once the predicate has held for `hold_s`, then waits for release."""

    def __init__(self, name, predicate: Callable[[PoseCtx], bool],
                 hold_s: float = 0.15, cooldown: float = 0.7):
        self.name, self.predicate, self.hold_s, self.cooldown = \
            name, predicate, hold_s, cooldown
        self._held = 0.0
        self._armed = True
        self._last = -1e9

    def update(self, c):
        on = bool(self.predicate(c))
        if on:
            self._held += c.feats.dt
        else:
            self._held = 0.0
            self._armed = True            # released -> can fire again
        if (on and self._held >= self.hold_s and self._armed
                and c.t - self._last >= self.cooldown):
            self._armed = False
            self._last = c.t
            return GestureEvent(self.name, c.t, 1.0, "pose")
        return None


class HeuristicMotion(Gesture):
    """Fires on a rising edge of a scalar trigger() above 0, with a refractory."""

    def __init__(self, name, trigger: Callable[[PoseCtx], float], cooldown: float = 0.5):
        self.name, self.trigger, self.cooldown = name, trigger, cooldown
        self._last = -1e9

    def update(self, c):
        s = float(self.trigger(c))
        if s > 0 and c.t - self._last >= self.cooldown:
            self._last = c.t
            return GestureEvent(self.name, c.t, min(1.0, s), "motion")
        return None


class DTWGesture(Gesture):
    """Match the recent motion window against a recorded reference via DTW."""

    def __init__(self, name, template: np.ndarray, joints: Sequence[str],
                 threshold: float, cooldown: float = 0.8):
        self.name = name
        self.template = _whiten(template)       # (M, D)
        self.joints = list(joints)
        self.idx = [JOINT_INDEX[j] for j in joints]
        self.threshold = threshold
        self.cooldown = cooldown
        self._last = -1e9

    def feat(self, xyz) -> np.ndarray:
        return xyz[self.idx, :2].reshape(-1)     # x,y of the chosen joints

    def update_window(self, window: np.ndarray, t: float) -> Optional[GestureEvent]:
        if window.shape[0] < 4 or t - self._last < self.cooldown:
            return None
        dist = dtw_distance(_whiten(window), self.template)
        if dist <= self.threshold:
            self._last = t
            return GestureEvent(self.name, t, 1.0 - dist / self.threshold, "motion")
        return None

    def update(self, c):                          # driven via recognizer window
        return None


# -- DTW -------------------------------------------------------------------

def _whiten(seq: np.ndarray) -> np.ndarray:
    """Translation+scale normalise a (T,D) trajectory so DTW compares *shape*."""
    seq = np.asarray(seq, float)
    seq = seq - seq.mean(0, keepdims=True)
    s = np.sqrt((seq ** 2).mean()) + 1e-6
    return seq / s


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised DTW distance between two (T,D) sequences (euclidean local cost)."""
    na, nb = len(a), len(b)
    D = np.full((na + 1, nb + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, na + 1):
        ai = a[i - 1]
        for j in range(1, nb + 1):
            cost = np.linalg.norm(ai - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[na, nb] / (na + nb))


# -- recognizer ------------------------------------------------------------

class GestureRecognizer:
    def __init__(self, gestures: List[Gesture], dtw_window_s: float = 1.2,
                 fps: float = 30.0):
        self.gestures = gestures
        self._dtw = [g for g in gestures if isinstance(g, DTWGesture)]
        self._win = max(8, int(dtw_window_s * fps))
        self._prev_xyz: Optional[np.ndarray] = None
        self._hist: List[np.ndarray] = []        # rolling raw-xyz buffer for DTW
        self._vert_hi = 1e-6                      # running standing verticality

    def update(self, frame, feats) -> List[GestureEvent]:
        if not getattr(frame, "raw_present", True):
            return []
        xyz = frame.xyz
        dt = max(feats.dt, 1e-3)
        vel = (xyz - self._prev_xyz) / dt if self._prev_xyz is not None else np.zeros_like(xyz)
        self._prev_xyz = xyz.copy()
        self._vert_hi = max(feats.verticality, self._vert_hi * 0.9995)
        vnorm = feats.verticality / self._vert_hi
        c = PoseCtx(xyz, vel, feats, frame.t, vnorm=vnorm)

        events: List[GestureEvent] = []
        for g in self.gestures:
            ev = g.update(c)
            if ev is not None:
                events.append(ev)

        if self._dtw:
            self._hist.append(xyz.copy())
            if len(self._hist) > self._win:
                self._hist.pop(0)
            buf = np.stack(self._hist)            # (T, J, 3)
            for g in self._dtw:
                win = buf[:, g.idx, :2].reshape(len(buf), -1)
                ev = g.update_window(win, frame.t)
                if ev is not None:
                    events.append(ev)
        return events


def record_template(frames_xyz: Sequence[np.ndarray], joints: Sequence[str],
                    name: str, threshold: float = 0.9) -> DTWGesture:
    """Build a DTWGesture from a recorded snippet of skeleton frames."""
    idx = [JOINT_INDEX[j] for j in joints]
    tmpl = np.stack([f[idx, :2].reshape(-1) for f in frames_xyz])
    return DTWGesture(name, tmpl, joints, threshold)


# -- built-in library ------------------------------------------------------
# Torso-normalised skeleton: hips at 0, torso = 1, shoulders ~+1, head ~+1.35,
# wrists-down ~+0.1, ankles ~-2. We only use joint *relationships* (robust to
# the hip-centring, which removes global translation -- so e.g. a jump, which is
# pure vertical translation, is intentionally not in this set).

def _hands_up(c):
    return c.p("l_wrist")[1] > c.p("nose")[1] and c.p("r_wrist")[1] > c.p("nose")[1]


def _raise(side):
    other = "r" if side == "l" else "l"
    def pred(c):
        up = c.p(f"{side}_wrist")[1] > c.p(f"{side}_shoulder")[1] + 0.4
        down = c.p(f"{other}_wrist")[1] < c.p(f"{other}_shoulder")[1]
        return up and down
    return pred


def _t_pose(c):
    ls, rs = c.p("l_shoulder"), c.p("r_shoulder")
    lw, rw = c.p("l_wrist"), c.p("r_wrist")
    level = abs(lw[1] - ls[1]) < 0.45 and abs(rw[1] - rs[1]) < 0.45
    wide = abs(lw[0] - rw[0]) > 1.8              # arms spread ~3 torso wide
    return level and wide


def _squat(c):
    return c.vnorm < 0.8                         # crouched below standing height


def _arms_crossed(c):
    lw, rw = c.p("l_wrist"), c.p("r_wrist")
    swapped = (lw[0] - rw[0]) > 0.15             # wrists crossed past each other
    chest = 0.5 * (c.p("l_shoulder")[1] + 0.0)
    return swapped and lw[1] < chest + 0.6 and rw[1] < chest + 0.6


def _clap(c):
    lw, rw = c.p("l_wrist"), c.p("r_wrist")
    close = np.linalg.norm(lw - rw) < 0.5
    not_crossed = (rw[0] - lw[0]) > -0.1          # meeting, not crossed (vs ARMS CROSSED)
    closing = float((c.v("l_wrist") - c.v("r_wrist")) @ (rw - lw)) > 1.5
    return 1.0 if (close and closing and not_crossed) else 0.0


def _punch(c):
    # a one-arm horizontal jab: near shoulder height, fast & horizontal, the
    # other arm tucked (distinguishes it from T-pose / hands-up extensions).
    for w, s, o, os in (("r_wrist", "r_shoulder", "l_wrist", "l_shoulder"),
                        ("l_wrist", "l_shoulder", "r_wrist", "r_shoulder")):
        reach, vel = c.p(w) - c.p(s), c.v(w)
        horiz = float(np.hypot(reach[0], reach[2]))
        other = abs(c.p(o)[0] - c.p(os)[0])
        if (abs(reach[1]) < 0.45 and horiz > 0.9 and np.linalg.norm(vel) > 4.0
                and abs(vel[0]) > abs(vel[1]) and float(vel @ reach) > 0 and other < 0.8):
            return 1.0
    return 0.0


def default_gestures(cooldown: float = 0.7) -> List[Gesture]:
    return [
        StaticPose("HANDS UP", _hands_up, hold_s=0.12, cooldown=cooldown),
        StaticPose("RAISE LEFT", _raise("l"), hold_s=0.12, cooldown=cooldown),
        StaticPose("RAISE RIGHT", _raise("r"), hold_s=0.12, cooldown=cooldown),
        StaticPose("T-POSE", _t_pose, hold_s=0.30, cooldown=cooldown),
        StaticPose("SQUAT", _squat, hold_s=0.20, cooldown=cooldown),
        StaticPose("ARMS CROSSED", _arms_crossed, hold_s=0.25, cooldown=cooldown),
        HeuristicMotion("CLAP", _clap, cooldown=0.5),
        HeuristicMotion("PUNCH", _punch, cooldown=0.4),
    ]
