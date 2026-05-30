"""Movement is a stack, not a signal.

Raw joints are the wrong abstraction. We decompose by timescale so each lane
can be routed to the musical layer that moves at the same rate:

  fast (ms)    -> velocity / acceleration / jerk / kinetic energy
  mid (beat)   -> the movement-energy envelope the oscillator entrains to
  slow (phrase)-> posture: COM height, openness, symmetry, verticality
  events       -> stomp, thrust, direction reversal, freeze
  signature    -> running statistics for personalisation

Everything here is per-frame and stateful; feed frames in order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .filters import EMA, OneEuroFilter
from .pose import JOINT_INDEX, JOINTS, MASS_VEC, PoseFrame


@dataclass
class Features:
    t: float
    dt: float
    # fast lane
    velocity: np.ndarray              # (J,3)
    speed: np.ndarray                 # (J,)
    accel_mag: np.ndarray             # (J,)
    jerk_mag: np.ndarray              # (J,)
    kinetic_energy: float             # scalar, mass-weighted
    # mid lane
    energy_env: float                 # smoothed energy envelope (oscillator food)
    # slow lane (posture)
    com: np.ndarray                   # (3,)
    com_height: float                 # COM height above the feet (slow, posture)
    bounce: float                     # COM height above feet (fast, the beat carrier)
    openness: float                   # limb extension / bounding spread
    symmetry: float                   # 0 = asymmetric, 1 = symmetric
    asymmetry_lr: float               # signed L-R activity (-1..1)
    verticality: float                # upright vs crouched
    # events (booleans / impulses this frame)
    events: Dict[str, float] = field(default_factory=dict)
    present: bool = True


class FeatureExtractor:
    def __init__(self, fps: float = 30.0):
        self.fps = fps
        # Fast lane: light smoothing only.
        self._pos_filt = OneEuroFilter(min_cutoff=2.0, beta=0.6, d_cutoff=2.0)
        # Envelope smoothing for the beat lane (a few hundred ms).
        self._env = EMA(tau=0.12)
        self._com_height = EMA(tau=0.5)   # slow lane buffers harder
        self._openness = EMA(tau=0.5)
        # history
        self._prev_pos: Optional[np.ndarray] = None
        self._prev_vel: Optional[np.ndarray] = None
        self._prev_accel: Optional[np.ndarray] = None
        self._prev_t: Optional[float] = None
        self._prev_com_vel: Optional[np.ndarray] = None
        # event bookkeeping
        self._still_time = 0.0
        self._energy_hist = []          # short ring for reversal/thrust baselines
        # running signature stats
        self.signature = SignatureAccumulator()

    def _openness_metric(self, xyz: np.ndarray) -> float:
        # Mean distance of the extremities from the COM, plus bounding spread.
        com = (xyz * MASS_VEC[:, None]).sum(0)
        ext = ["l_wrist", "r_wrist", "l_ankle", "r_ankle", "nose"]
        d = np.mean([np.linalg.norm(xyz[JOINT_INDEX[j]] - com) for j in ext])
        spread = float(np.ptp(xyz[:, 0]) + np.ptp(xyz[:, 1]))
        return float(0.5 * d + 0.25 * spread)

    def update(self, frame: PoseFrame) -> Features:
        t = frame.t
        if self._prev_t is None:
            dt = 1.0 / self.fps
        else:
            dt = max(t - self._prev_t, 1e-3)

        if not frame.raw_present:
            # No body: emit a quiet, "present=False" feature frame.
            self._prev_t = t
            return self._empty(t, dt)

        pos = self._pos_filt(frame.xyz, dt)  # (J,3)

        if self._prev_pos is None:
            vel = np.zeros_like(pos)
        else:
            vel = (pos - self._prev_pos) / dt
        if self._prev_vel is None:
            accel = np.zeros_like(pos)
        else:
            accel = (vel - self._prev_vel) / dt
        if self._prev_accel is None:
            jerk = np.zeros_like(pos)
        else:
            jerk = (accel - self._prev_accel) / dt

        speed = np.linalg.norm(vel, axis=1)
        accel_mag = np.linalg.norm(accel, axis=1)
        jerk_mag = np.linalg.norm(jerk, axis=1)
        kinetic = float(0.5 * (MASS_VEC * speed ** 2).sum())
        energy_env = float(self._env(kinetic, dt))

        com = (pos * MASS_VEC[:, None]).sum(0)
        # Height above the feet: survives hip-centring, so a knee-flex bounce is
        # visible here (unlike COM measured relative to the hips).
        ankle_y = 0.5 * (pos[JOINT_INDEX["l_ankle"], 1] + pos[JOINT_INDEX["r_ankle"], 1])
        height = float(com[1] - ankle_y)
        com_height = float(self._com_height(height, dt))   # slow: posture
        bounce = height                                    # fast: beat carrier
        openness = float(self._openness(self._openness_metric(pos), dt))

        # symmetry / asymmetry from L vs R limb speed
        l_idx = [JOINT_INDEX[j] for j in JOINTS if j.startswith("l_")]
        r_idx = [JOINT_INDEX[j] for j in JOINTS if j.startswith("r_")]
        l_act = float(speed[l_idx].mean())
        r_act = float(speed[r_idx].mean())
        tot = l_act + r_act + 1e-6
        asym = (l_act - r_act) / tot
        symmetry = 1.0 - abs(asym)

        shoulder = 0.5 * (pos[JOINT_INDEX["l_shoulder"]] + pos[JOINT_INDEX["r_shoulder"]])
        ankle = 0.5 * (pos[JOINT_INDEX["l_ankle"]] + pos[JOINT_INDEX["r_ankle"]])
        verticality = float((shoulder[1] - ankle[1]))  # taller stack -> upright

        events = self._detect_events(pos, vel, accel, kinetic, energy_env, dt)

        feats = Features(
            t=t, dt=dt, velocity=vel, speed=speed, accel_mag=accel_mag,
            jerk_mag=jerk_mag, kinetic_energy=kinetic, energy_env=energy_env,
            com=com, com_height=com_height, bounce=bounce, openness=openness,
            symmetry=symmetry, asymmetry_lr=float(np.clip(asym, -1, 1)),
            verticality=verticality, events=events, present=True,
        )

        self._prev_pos, self._prev_vel, self._prev_accel = pos, vel, accel
        self._prev_t = t
        self._prev_com_vel = (com - (self._prev_com if hasattr(self, "_prev_com") else com)) / dt
        self._prev_com = com
        self.signature.update(feats)
        return feats

    def _detect_events(self, pos, vel, accel, kinetic, env, dt) -> Dict[str, float]:
        ev: Dict[str, float] = {}
        com_vy = float((vel * MASS_VEC[:, None]).sum(0)[1])
        com_ay = float((accel * MASS_VEC[:, None]).sum(0)[1])

        # Stomp: sharp downward COM acceleration after descent (landing).
        if com_ay > 8.0 and com_vy < 0.2:
            ev["stomp"] = float(np.clip(com_ay / 20.0, 0, 1))

        # Thrust: sudden whole-body extension (kinetic spike vs recent baseline).
        base = np.mean(self._energy_hist) if self._energy_hist else kinetic
        if kinetic > 3.0 * (base + 1e-3) and kinetic > 0.5:
            ev["thrust"] = float(np.clip(kinetic / (base + 1e-3) / 6.0, 0, 1))
        self._energy_hist.append(kinetic)
        if len(self._energy_hist) > 30:
            self._energy_hist.pop(0)

        # Direction reversal: dominant-limb velocity sign flip with magnitude.
        wr = JOINT_INDEX["r_wrist"]
        if self._prev_vel is not None:
            prev = self._prev_vel[wr]
            cur = vel[wr]
            flip = np.dot(prev, cur)
            if flip < 0 and np.linalg.norm(cur) > 0.5:
                ev["reversal"] = float(np.clip(np.linalg.norm(cur) / 3.0, 0, 1))

        # Freeze: sustained low energy. Tracked, exposed as a level (0..1).
        if env < 0.02:
            self._still_time += dt
        else:
            self._still_time = 0.0
        ev["freeze"] = float(np.clip(self._still_time / 1.0, 0, 1))
        return ev

    def _empty(self, t: float, dt: float) -> Features:
        J = len(JOINTS)
        self._still_time += dt
        return Features(
            t=t, dt=dt, velocity=np.zeros((J, 3)), speed=np.zeros(J),
            accel_mag=np.zeros(J), jerk_mag=np.zeros(J), kinetic_energy=0.0,
            energy_env=0.0, com=np.zeros(3), com_height=0.0, bounce=0.0,
            openness=0.0,
            symmetry=1.0, asymmetry_lr=0.0, verticality=0.0,
            events={"freeze": 1.0}, present=False,
        )


class SignatureAccumulator:
    """Running statistics that describe *how this person moves*."""

    def __init__(self):
        self.n = 0
        self.energy_sum = 0.0
        self.energy_sq = 0.0
        self.openness_sum = 0.0
        self.jerk_sum = 0.0
        self.com_min = np.inf
        self.com_max = -np.inf
        self.energy_max = 0.0
        self.openness_min = np.inf
        self.openness_max = -np.inf

    def update(self, f: Features):
        self.n += 1
        self.energy_sum += f.energy_env
        self.energy_sq += f.energy_env ** 2
        self.openness_sum += f.openness
        self.jerk_sum += float(f.jerk_mag.mean())
        self.com_min = min(self.com_min, f.com_height)
        self.com_max = max(self.com_max, f.com_height)
        self.energy_max = max(self.energy_max, f.energy_env)
        self.openness_min = min(self.openness_min, f.openness)
        self.openness_max = max(self.openness_max, f.openness)

    def summary(self) -> dict:
        n = max(self.n, 1)
        mean_e = self.energy_sum / n
        var_e = max(self.energy_sq / n - mean_e ** 2, 0.0)
        return {
            "n": self.n,
            "energy_mean": mean_e,
            "energy_std": var_e ** 0.5,
            "energy_max": self.energy_max,
            "openness_mean": self.openness_sum / n,
            "openness_range": (self.openness_min, self.openness_max),
            "jerk_mean": self.jerk_sum / n,
            "com_range": (self.com_min, self.com_max),
        }
